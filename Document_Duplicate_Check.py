import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import hashlib
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import re
from PyPDF2 import PdfReader
import docx
import pytesseract
from PIL import Image
import mimetypes
import joblib
import faiss
from pymongo import MongoClient
from bson import ObjectId
import datetime
import pickle

class DuplicateDetectionModel:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_gpu=False, similarity_threshold=0.85,
                 mongo_uri='mongodb://localhost:27017', db_name='document_db'):
        """
        문서 중복 검사를 위한 머신러닝 모델 (MongoDB 통합)
        
        Args:
            model_name (str): 사용할 임베딩 모델 이름
            use_gpu (bool): GPU 사용 여부
            similarity_threshold (float): 중복으로 판단할 유사도 임계값 (0~1)
            mongo_uri (str): MongoDB 연결 URI
            db_name (str): 사용할 데이터베이스 이름
        """
        self.similarity_threshold = similarity_threshold
        
        # MongoDB 연결 설정
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.documents_collection = self.db['documents']
        self.embeddings_collection = self.db['embeddings']
        self.file_hashes_collection = self.db['file_hashes']
        
        # 임베딩 모델 로드
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
        
        # FAISS 인덱스 초기화
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.vector_dimension)  # 내적(코사인 유사도용)
        
        # 기존 임베딩 데이터 로드
        self._load_embeddings_from_mongodb()
    
    def _load_embeddings_from_mongodb(self):
        """MongoDB에서 임베딩 데이터 로드하여 FAISS 인덱스에 추가"""
        embeddings_docs = list(self.embeddings_collection.find({}))
        
        if len(embeddings_docs) > 0:
            # 문서 ID 목록 구성
            self.document_ids = [str(doc['document_id']) for doc in embeddings_docs]
            
            # 임베딩 벡터 로드하여 FAISS 인덱스에 추가
            embeddings = np.array([doc['embedding'] for doc in embeddings_docs], dtype=np.float32)
            if len(embeddings) > 0:
                self.index.add(embeddings)
                print(f"Loaded {len(embeddings)} embeddings from MongoDB")
        else:
            self.document_ids = []
            print("No embeddings found in MongoDB")
    
    def _extract_text(self, file_path):
        """파일에서 텍스트 추출"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type is None:
            # 확장자로 파일 타입 추정
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                mime_type = 'application/pdf'
            elif ext in ['.doc', '.docx']:
                mime_type = 'application/msword'
            elif ext in ['.txt', '.md']:
                mime_type = 'text/plain'
            elif ext in ['.jpg', '.jpeg', '.png']:
                mime_type = 'image/jpeg'
        
        # 파일 타입에 따라 텍스트 추출
        if mime_type and mime_type.startswith('text/'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
            
        elif mime_type == 'application/pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        elif mime_type and (mime_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml') or mime_type == 'application/msword'):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
            
        elif mime_type and mime_type.startswith('image/'):
            # 이미지에서 OCR로 텍스트 추출
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
            
        return ""
    
    def _compute_file_hash(self, file_path):
        """파일의 MD5 해시 계산"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # 64kb 단위로 읽기
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _preprocess_text(self, text):
        """텍스트 전처리"""
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _compute_embedding(self, text):
        """텍스트 임베딩 계산"""
        if not text:
            return np.zeros(self.vector_dimension)
        return self.model.encode(text, convert_to_numpy=True)
    
    def add_document(self, file_path, metadata=None):
        """
        문서 추가 및 중복 검사
        
        Args:
            file_path (str): 파일 경로
            metadata (dict, optional): 문서 관련 메타데이터
            
        Returns:
            dict: 중복 검사 결과
        """
        # 파일 해시 확인
        file_hash = self._compute_file_hash(file_path)
        
        # MongoDB에서 해시로 중복 확인
        existing_hash = self.file_hashes_collection.find_one({"hash": file_hash})
        if existing_hash:
            # 완전히 동일한 파일을 발견
            duplicate_doc = self.documents_collection.find_one({"_id": existing_hash["document_id"]})
            return {
                'is_duplicate': True,
                'duplicate_type': 'exact',
                'duplicate_id': str(existing_hash["document_id"]),
                'document_title': duplicate_doc.get('title', 'Unknown'),
                'similarity': 1.0
            }
        
        # 텍스트 추출 및 임베딩 계산
        text = self._extract_text(file_path)
        processed_text = self._preprocess_text(text)
        embedding = self._compute_embedding(processed_text)
        
        # 벡터 정규화 (코사인 유사도를 위해)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        embedding_for_faiss = normalized_embedding.reshape(1, -1).astype(np.float32)
        
        # 유사 문서 검색 (기존 문서가 있을 경우)
        similar_doc_info = None
        if len(self.document_ids) > 0:
            # FAISS로 유사도 검색
            D, I = self.index.search(embedding_for_faiss, min(5, len(self.document_ids)))
            
            # 유사도가 임계값을 넘는지 확인
            if D[0][0] > self.similarity_threshold:
                most_similar_idx = I[0][0]
                similar_doc_id = self.document_ids[most_similar_idx]
                
                # 유사 문서 정보 가져오기
                similar_doc = self.documents_collection.find_one({"_id": ObjectId(similar_doc_id)})
                
                similar_doc_info = {
                    'is_duplicate': True,
                    'duplicate_type': 'similar',
                    'duplicate_id': similar_doc_id,
                    'document_title': similar_doc.get('title', 'Unknown'),
                    'similarity': float(D[0][0])
                }
        
        # 중복이 아니거나, 사용자가 중복을 무시하고 추가하기를 원할 경우
        if similar_doc_info is None:
            # 파일명을 기본 제목으로 사용
            file_name = os.path.basename(file_path)
            
            # 문서 메타데이터 준비
            if metadata is None:
                metadata = {}
                
            # MongoDB에 문서 정보 저장
            document_data = {
                "title": metadata.get("title", file_name),
                "file_path": file_path,
                "file_type": mimetypes.guess_type(file_path)[0],
                "text_content": text[:1000],  # 전체 텍스트 저장하지 않고 일부만 저장 (필요시 변경)
                "created_at": datetime.datetime.now(),
                "metadata": metadata
            }
            
            document_id = self.documents_collection.insert_one(document_data).inserted_id
            
            # 파일 해시 저장
            self.file_hashes_collection.insert_one({
                "hash": file_hash,
                "document_id": document_id,
                "created_at": datetime.datetime.now()
            })
            
            # 임베딩 저장
            embedding_data = {
                "document_id": document_id,
                "embedding": normalized_embedding.tolist(),
                "created_at": datetime.datetime.now()
            }
            self.embeddings_collection.insert_one(embedding_data)
            
            # FAISS 인덱스 업데이트
            self.index.add(embedding_for_faiss)
            self.document_ids.append(str(document_id))
            
            return {
                'is_duplicate': False,
                'document_id': str(document_id),
                'title': document_data["title"]
            }
        
        return similar_doc_info
    
    def get_document_by_id(self, document_id):
        """ID로 문서 정보 조회"""
        return self.documents_collection.find_one({"_id": ObjectId(document_id)})
    
    def get_most_similar_documents(self, file_path=None, text=None, document_id=None, top_k=5):
        """
        가장 유사한 문서 k개 찾기
        
        Args:
            file_path (str, optional): 파일 경로
            text (str, optional): 직접 비교할 텍스트
            document_id (str, optional): 이미 저장된 문서 ID
            top_k (int): 반환할 유사 문서 수
            
        Returns:
            list: 유사 문서 목록
        """
        if len(self.document_ids) == 0:
            return []
        
        embedding = None
        
        if document_id:
            # 문서 ID로 임베딩 찾기
            embedding_doc = self.embeddings_collection.find_one({"document_id": ObjectId(document_id)})
            if embedding_doc:
                embedding = np.array(embedding_doc["embedding"], dtype=np.float32)
        
        elif file_path:
            # 파일에서 텍스트 추출
            text = self._extract_text(file_path)
            text = self._preprocess_text(text)
            embedding = self._compute_embedding(text)
        
        elif text:
            # 직접 제공된 텍스트 사용
            text = self._preprocess_text(text)
            embedding = self._compute_embedding(text)
        
        else:
            return []
        
        # 벡터 정규화
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # 유사한 문서 검색
        k = min(top_k, len(self.document_ids))
        D, I = self.index.search(embedding, k)
        
        results = []
        for i in range(k):
            doc_id = ObjectId(self.document_ids[I[0][i]])
            doc = self.documents_collection.find_one({"_id": doc_id})
            
            if doc:
                results.append({
                    'document_id': str(doc_id),
                    'title': doc.get('title', 'Unknown'),
                    'similarity': float(D[0][i]),
                    'created_at': doc.get('created_at', None),
                    'file_type': doc.get('file_type', None),
                    'metadata': doc.get('metadata', {})
                })
        
        return results
    
    def delete_document(self, document_id):
        """문서 및 관련 데이터 삭제"""
        # ObjectId로 변환
        doc_id = ObjectId(document_id)
        
        # 문서 존재 확인
        doc = self.documents_collection.find_one({"_id": doc_id})
        if not doc:
            return {"success": False, "message": "Document not found"}
        
        # 해시 정보 삭제
        self.file_hashes_collection.delete_many({"document_id": doc_id})
        
        # 임베딩 정보 찾기
        embedding_doc = self.embeddings_collection.find_one({"document_id": doc_id})
        
        # 문서 삭제
        self.documents_collection.delete_one({"_id": doc_id})
        
        # 임베딩 삭제
        self.embeddings_collection.delete_one({"document_id": doc_id})
        
        # FAISS 인덱스 및 document_ids 업데이트
        # (FAISS는 삭제를 직접 지원하지 않으므로 인덱스를 재구성)
        if embedding_doc:
            # 문서 ID 목록에서 제거
            try:
                idx = self.document_ids.index(str(doc_id))
                self.document_ids.pop(idx)
                
                # 새 인덱스 구성 (모든 임베딩 재로드)
                self._rebuild_faiss_index()
            except ValueError:
                pass  # document_ids에 없는 경우
        
        return {"success": True, "message": "Document deleted successfully"}
    
    def _rebuild_faiss_index(self):
        """FAISS 인덱스 재구성"""
        # 새 인덱스 생성
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        # MongoDB에서 모든 임베딩 가져오기
        embeddings_docs = list(self.embeddings_collection.find({}))
        
        if len(embeddings_docs) > 0:
            # 문서 ID 목록 재구성
            self.document_ids = [str(doc['document_id']) for doc in embeddings_docs]
            
            # 임베딩 벡터 로드하여 FAISS 인덱스에 추가
            embeddings = np.array([doc['embedding'] for doc in embeddings_docs], dtype=np.float32)
            if len(embeddings) > 0:
                self.index.add(embeddings)
    
    def categorize_document(self, document_id, categories):
        """
        문서 카테고리 지정
        
        Args:
            document_id (str): 문서 ID
            categories (list): 카테고리 목록
            
        Returns:
            dict: 업데이트 결과
        """
        try:
            result = self.documents_collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"categories": categories, "updated_at": datetime.datetime.now()}}
            )
            return {
                "success": result.modified_count > 0,
                "message": "Categories updated" if result.modified_count > 0 else "No changes made"
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def search_documents(self, query, filter_criteria=None, limit=10):
        """
        문서 검색 (텍스트 기반)
        
        Args:
            query (str): 검색어
            filter_criteria (dict, optional): 필터 조건
            limit (int): 최대 결과 수
            
        Returns:
            list: 검색 결과
        """
        # 검색어 처리
        processed_query = self._preprocess_text(query)
        
        # 임베딩 계산
        query_embedding = self._compute_embedding(processed_query)
        normalized_embedding = query_embedding / np.linalg.norm(query_embedding)
        embedding_for_search = normalized_embedding.reshape(1, -1).astype(np.float32)
        
        # 유사한 문서 검색
        if len(self.document_ids) == 0:
            return []
        
        k = min(limit * 2, len(self.document_ids))  # 필터링 고려하여 더 많이 가져옴
        D, I = self.index.search(embedding_for_search, k)
        
        # 결과 가공
        candidate_docs = []
        for i in range(len(I[0])):
            doc_id = ObjectId(self.document_ids[I[0][i]])
            
            # 필터 조건 적용
            query_filter = {"_id": doc_id}
            if filter_criteria:
                for key, value in filter_criteria.items():
                    query_filter[key] = value
            
            doc = self.documents_collection.find_one(query_filter)
            if doc:
                candidate_docs.append({
                    'document_id': str(doc_id),
                    'title': doc.get('title', 'Unknown'),
                    'similarity': float(D[0][i]),
                    'created_at': doc.get('created_at', None),
                    'categories': doc.get('categories', []),
                    'metadata': doc.get('metadata', {})
                })
        
        # 결과 제한
        return candidate_docs[:limit]
