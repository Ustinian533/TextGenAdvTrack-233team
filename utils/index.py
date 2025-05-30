# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=16):
        # if n_subquantizers > 0:
        #     self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        # else:
        self.vector_sz = vector_sz
        self.index = self._create_sharded_index()
        self.index_id_to_db_id = []
        self.label_dict = {}
        # self.index = faiss.IndexFlatIP(vector_sz)

        # self.index = faiss.index_cpu_to_all_gpus(self.index)
        # #self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        # self.index_id_to_db_id = []
        # self.label_dict = {}

    def _create_sharded_index(self):
        # 获取可用的GPU数量
        ngpu = faiss.get_num_gpus()
        # 创建IndexShards对象，参数successive_ids=True确保ID在全局范围内唯一
        index = faiss.IndexShards(self.vector_sz, True, True)
        # 为每个GPU创建一个子索引并添加到IndexShards
        for i in range(ngpu):
            # 创建标准的GPU资源对象
            res = faiss.StandardGpuResources()
            # 配置GPU索引
            flat_config = faiss.GpuIndexFlatConfig()
            # flat_config.useFloat16 = True  # 使用半精度浮点数以节省显存
            flat_config.device = i  # 指定GPU设备
            # 创建GPU索引
            sub_index = faiss.GpuIndexFlatIP(res, self.vector_sz, flat_config)
            # 将子索引添加到IndexShards
            index.add_shard(sub_index)
        return index

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f'Total data indexed {self.index.ntotal}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 8) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            db_labels = [[self.label_dict[self.index_id_to_db_id[i]] for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i],db_labels[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)

    def reset(self):
        self.index.reset()
        self.index_id_to_db_id = []
        print(f'Index reset, total data indexed {self.index.ntotal}')