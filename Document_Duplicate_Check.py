# pydantic
# fastapi
# mongoDB

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

class DuplicateDetectionModel:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_gpu=False, similarity_threshold=0.85):
        """
        문서 중복 검사를 위한 머신러닝 모델
        
        Args:
            model_name (str): 사용할 임베딩 모델 이름
            use_gpu (bool): GPU 사용 여부
            similarity_threshold (float): 중복으로 판단할 유사도 임계값 (0~1)
        """
        self.similarity_threshold = similarity_threshold
        
        # 파일 해시값을 저장할 딕셔너리
        self.file_hashes = {}
        
        # 문서 임베딩을 저장할 데이터베이스
        self.embeddings_db = None
        self.document_ids = []
        
        # 임베딩 모델 로드
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
        
        # FAISS 인덱스 초기화
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.vector_dimension)  # 내적(코사인 유사도용)
    
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
        if mime_type.startswith('text/'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
            
        elif mime_type == 'application/pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        elif mime_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml') or mime_type == 'application/msword':
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
            
        elif mime_type.startswith('image/'):
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
    
    def add_document(self, file_path, document_id):
        """문서 추가"""
        # 파일 해시 확인
        file_hash = self._compute_file_hash(file_path)
        
        # 완전히 동일한 파일인지 해시로 먼저 확인
        if file_hash in self.file_hashes:
            return {
                'is_duplicate': True,
                'duplicate_type': 'exact',
                'duplicate_id': self.file_hashes[file_hash],
                'similarity': 1.0
            }
        
        # 새 파일 해시 저장
        self.file_hashes[file_hash] = document_id
        
        # 텍스트 추출 및 임베딩 계산
        text = self._extract_text(file_path)
        text = self._preprocess_text(text)
        embedding = self._compute_embedding(text)
        
        # 벡터 정규화 (코사인 유사도를 위해)
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # 기존 문서가 있을 경우 유사도 확인
        if len(self.document_ids) > 0:
            # FAISS로 빠른 유사도 검색
            D, I = self.index.search(embedding, min(5, len(self.document_ids)))
            
            # 유사도가 임계값을 넘는지 확인
            if D[0][0] > self.similarity_threshold:
                most_similar_idx = I[0][0]
                return {
                    'is_duplicate': True,
                    'duplicate_type': 'similar',
                    'duplicate_id': self.document_ids[most_similar_idx],
                    'similarity': float(D[0][0])
                }
        
        # 중복이 아니면 DB에 추가
        self.index.add(embedding)
        self.document_ids.append(document_id)
        
        return {
            'is_duplicate': False,
            'document_id': document_id
        }
    
    def save_model(self, model_path, index_path):
        """모델 저장"""
        # 모델 상태 저장
        model_state = {
            'file_hashes': self.file_hashes,
            'document_ids': self.document_ids,
            'similarity_threshold': self.similarity_threshold,
            'vector_dimension': self.vector_dimension
        }
        joblib.dump(model_state, model_path)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, index_path)
    
    def load_model(self, model_path, index_path):
        """모델 로드"""
        # 모델 상태 로드
        model_state = joblib.load(model_path)
        self.file_hashes = model_state['file_hashes']
        self.document_ids = model_state['document_ids']
        self.similarity_threshold = model_state['similarity_threshold']
        self.vector_dimension = model_state['vector_dimension']
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(index_path)
    
    def get_most_similar_documents(self, file_path, top_k=5):
        """가장 유사한 문서 k개 찾기"""
        # 텍스트 추출 및 임베딩 계산
        text = self._extract_text(file_path)
        text = self._preprocess_text(text)
        embedding = self._compute_embedding(text)
        
        # 벡터 정규화
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # 유사한 문서 검색
        if len(self.document_ids) == 0:
            return []
        
        k = min(top_k, len(self.document_ids))
        D, I = self.index.search(embedding, k)
        
        results = []
        for i in range(k):
            results.append({
                'document_id': self.document_ids[I[0][i]],
                'similarity': float(D[0][i])
            })
        
        return results

# 사용 예시
if __name__ == "__main__":
    # 모델 초기화
    model = DuplicateDetectionModel(similarity_threshold=0.85)
    
    # 예시: 문서 추가 및 중복 확인
    doc1_result = model.add_document('c:/Users/Lenovo/Desktop/code/Git/AI-from-basic/Block_Chein/doc1.pdf', 'doc1')
    print(f"Document 1 result: {doc1_result}")
    
    doc2_result = model.add_document('c:/Users/Lenovo/Desktop/code/Git/AI-from-basic/Block_Chein/doc2.pdf', 'doc2')
    print(f"Document 2 result: {doc2_result}")
    
    # 유사 문서 찾기
    similar_docs = model.get_most_similar_documents('c:/Users/Lenovo/Desktop/code/Git/AI-from-basic/Block_Chein/doc3.pdf', top_k=3)
    print(f"Similar documents: {similar_docs}")
    
    # 모델 저장
    model.save_model('duplicate_model.pkl', 'duplicate_index.faiss')
    
    # 모델 로드
    new_model = DuplicateDetectionModel()
    new_model.load_model('duplicate_model.pkl', 'duplicate_index.faiss')