import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
import re
import joblib
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract
import mimetypes

class DocumentClassifier:
    def __init__(self, model_name='klue/bert-base', use_gpu=False):
        """
        문서 자동 분류를 위한 머신러닝 모델
        
        Args:
            model_name (str): 사용할 사전 학습 모델 이름
            use_gpu (bool): GPU 사용 여부
        """
        self.model_name = model_name
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # 고정된 카테고리 목록 (시험자료, 자소서, 레포트, 이력서)
        self.categories = ["시험자료", "자소서", "레포트", "이력서"]
        self.num_categories = len(self.categories)
        
        # 카테고리 인코더
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        
        # 토크나이저와 모델 초기화
        self.tokenizer = None
        self.model = None
        
        # 미세 조정된 모델이 있는지 확인
        self.is_finetuned = False
    
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
            return pytesseract.image_to_string(img, lang='kor+eng')  # 한국어 및 영어 OCR 지원
            
        return ""
    
    def _preprocess_text(self, text):
        """텍스트 전처리"""
        # 소문자 변환 (영어 텍스트의 경우)
        text = text.lower()
        # 특수문자 제거 (공백으로 대체)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 한글 문자 보존
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 너무 긴 텍스트는 잘라내기 (BERT 모델의 최대 입력 길이 제한 때문)
        max_len = 512
        if len(text.split()) > max_len:
            text = ' '.join(text.split()[:max_len])
            
        return text
    
    def _prepare_dataset(self, texts, labels=None):
        """데이터셋 준비"""
        if labels is not None:
            # 훈련용 데이터셋
            return Dataset.from_dict({
                'text': texts,
                'label': labels
            })
        else:
            # 추론용 데이터셋
            return Dataset.from_dict({
                'text': texts
            })
    
    def _tokenize_function(self, examples):
        """토크나이징 함수"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def train(self, train_data, validation_split=0.2, epochs=3):
        """모델 학습"""
        texts = []
        labels = []
        
        # 학습 데이터 준비
        for file_path, category in train_data:
            # 카테고리가 유효한지 확인
            if category not in self.categories:
                raise ValueError(f"카테고리 '{category}'는 유효하지 않습니다. 유효한 카테고리: {self.categories}")
                
            text = self._extract_text(file_path)
            text = self._preprocess_text(text)
            
            texts.append(text)
            labels.append(category)
        
        # 카테고리를 정수로 인코딩
        encoded_labels = self.label_encoder.transform(labels)
        
        # 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_categories,
        )
        
        # 훈련/검증 분할
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=validation_split, random_state=42, stratify=encoded_labels
        )
        
        # 데이터셋 준비
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)
        
        # 토큰화
        train_tokenized = train_dataset.map(self._tokenize_function, batched=True)
        val_tokenized = val_dataset.map(self._tokenize_function, batched=True)
        
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # 트레이너 초기화 및 학습
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
        )
        
        trainer.train()
        
        # 학습 완료 플래그 설정
        self.is_finetuned = True
        
        # 평가
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        return eval_results
    
    def predict(self, file_path):
        """문서 분류 예측"""
        if not self.is_finetuned:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 실행하세요.")
        
        # 텍스트 추출 및 전처리
        text = self._extract_text(file_path)
        text = self._preprocess_text(text)
        
        # 토큰화
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 예측
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # 가장 높은 확률의 클래스 및 확률값
            max_prob, predicted_class_idx = torch.max(probabilities, dim=1)
            
            predicted_class_idx = predicted_class_idx.cpu().item()
            prediction_probability = max_prob.cpu().item()
        
        # 예측된 클래스 이름
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # 각 카테고리 확률
        category_probs = {}
        for i, cat in enumerate(self.categories):
            category_probs[cat] = probabilities[0, i].cpu().item()
        
        return {
            'predicted_category': predicted_class,
            'confidence': prediction_probability,
            'category_probabilities': category_probs
        }
    
    def save_model(self, model_dir):
        """모델 저장"""
        if not self.is_finetuned:
            raise ValueError("저장할 모델이 없습니다. 먼저 train() 메서드를 실행하세요.")
        
        # 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 저장
        self.model.save_pretrained(f"{model_dir}/model")
        self.tokenizer.save_pretrained(f"{model_dir}/tokenizer")
        
        # 레이블 인코더 저장
        joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.pkl")
        
        # 설정 저장
        config = {
            'model_name': self.model_name,
            'num_categories': self.num_categories,
            'is_finetuned': self.is_finetuned,
            'categories': self.categories
        }
        
        joblib.dump(config, f"{model_dir}/config.pkl")
    
    def load_model(self, model_dir):
        """모델 로드"""
        # 설정 로드
        config = joblib.load(f"{model_dir}/config.pkl")
        self.model_name = config['model_name']
        self.num_categories = config['num_categories']
        self.is_finetuned = config['is_finetuned']
        self.categories = config['categories']
        
        # 레이블 인코더 로드
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
        
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"{model_dir}/model",
            num_labels=self.num_categories
        )
        
        # 디바이스 설정
        self.model.to(self.device)
    
    def extract_keywords(self, file_path, top_k=10):
        """문서에서 키워드 추출"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 텍스트 추출 및 전처리
        text = self._extract_text(file_path)
        text = self._preprocess_text(text)
        
        # 한국어 불용어 목록 (간단한 예시)
        korean_stopwords = ['이', '그', '저', '것', '및', '등', '를', '을', '에', '의', '은', '는', '이', '가', '와', '과', '으로', '로', '에서']
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=korean_stopwords)
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # 단어 및 TF-IDF 점수
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 단어-점수 쌍 생성 및 정렬
        word_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 키워드 반환
        return [
            {'keyword': word, 'score': float(score)} 
            for word, score in word_scores[:top_k]
        ]
    
    def get_category_features(self):
        """각 카테고리별 특징적인 키워드 및 패턴"""
        return {
            "시험자료": ["문제", "답안", "시험", "테스트", "객관식", "주관식", "보기", "과목", "정답", "채점"],
            "자소서": ["지원동기", "자기소개", "경력", "역량", "지원자", "지원", "성장과정", "장단점", "인재상", "목표"],
            "레포트": ["서론", "본론", "결론", "참고문헌", "연구", "분석", "조사", "인용", "자료", "논의"],
            "이력서": ["경력사항", "학력", "자격증", "연락처", "기술", "경험", "프로젝트", "근무", "담당업무", "성명"]
        }

# 사용 예시
if __name__ == "__main__":
    # 모델 초기화
    classifier = DocumentClassifier(model_name='klue/bert-base')
    
    # 훈련 데이터 예시 (파일경로, 카테고리)
    train_data = [
        ('./data/exam1.pdf', '시험자료'),
        ('./data/exam2.pdf', '시험자료'),
        ('./data/exam3.pdf', '시험자료'),
        ('./data/cover1.pdf', '자소서'),
        ('./data/cover2.pdf', '자소서'),
        ('./data/cover3.pdf', '자소서'),
        ('./data/report1.pdf', '레포트'),
        ('./data/report2.pdf', '레포트'),
        ('./data/report3.pdf', '레포트'),
        ('./data/resume1.pdf', '이력서'),
        ('./data/resume2.pdf', '이력서'),
        ('./data/resume3.pdf', '이력서'),
    ]
    
    # 모델 학습
    classifier.train(train_data, epochs=3)
    
    # 모델 저장
    classifier.save_model('korean_document_classifier')
    
    # 모델 로드
    new_classifier = DocumentClassifier()
    new_classifier.load_model('korean_document_classifier')
    
    # 문서 분류
    result = new_classifier.predict('./data/new_document.pdf')
    print(f"예측된 카테고리: {result['predicted_category']} (확신도: {result['confidence']:.2f})")
    print(f"각 카테고리별 확률:")
    for cat, prob in result['category_probabilities'].items():
        print(f"  - {cat}: {prob:.2f}")
    
    # 키워드 추출
    keywords = new_classifier.extract_keywords('./data/new_document.pdf')
    print(f"주요 키워드:")
    for kw in keywords:
        print(f"  - {kw['keyword']}: {kw['score']:.4f}")