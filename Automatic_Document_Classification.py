import os
import re
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings

# 필요한 라이브러리 존재 확인
def check_required_libraries():
    missing_libraries = []
    try:
        import pytesseract
    except ImportError:
        missing_libraries.append("pytesseract")
    
    try:
        import pdf2image
    except ImportError:
        missing_libraries.append("pdf2image")
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        missing_libraries.append("PyMuPDF")
    
    if missing_libraries:
        print("경고: 다음 라이브러리가 없어 일부 기능이 제한됩니다:")
        for lib in missing_libraries:
            print(f"- {lib}")
        print("설치 방법: pip install " + " ".join(missing_libraries))
        return False
    
    return True

class PDFClassifier:
    def __init__(self):
        # 문서 유형 정의
        self.document_types = ['자기소개서', '이력서', '학습자료', '레포트', '기타']
        
        # 각 문서 유형별 주요 키워드 정의
        self.keywords = {
            '자기소개서': [
                # 한국어 키워드
                '지원동기', '자기소개', '성장과정', '장단점', '지원이유', '역량', '목표', '비전', '성취', '자소서', '지원원'
                # 영어 키워드
                'personal statement', 'statement of purpose', 'motivation letter', 'cover letter', 
                'self introduction', 'background', 'achievements', 'strengths', 'weaknesses', 
                'career goal', 'objective', 'vision', 'why I want', 'reason for applying'
            ],
            '이력서': [
                # 한국어 키워드
                '학력', '경력', '자격증', '기술', '연락처', '프로젝트', '업무', '성과', '경험', '담당업무', '인적사항',
                # 영어 키워드
                'resume', 'curriculum vitae', 'CV', 'education', 'experience', 'certificate', 
                'skills', 'contact', 'project', 'work history', 'achievements', 'responsibilities',
                'personal information', 'employment history', 'qualification', 'professional experience'
            ],
            '학습자료': [
                # 한국어 키워드
                '학습', '교육', '교재', '강의', '이론', '개념', '설명', '예제', '연습문제', '참고', '실습', 
                '교과서', '문제', '구하시오' , '구해라'
                # 영어 키워드
                'lecture', 'course material', 'textbook', 'learning', 'education', 'theory', 
                'concept', 'example', 'exercise', 'practice', 'study guide', 'reference material',
                'handbook', 'tutorial', 'lesson', 'syllabus', 'curriculum', 'workbook', 'prove',
                'solve', 'proof'
            ],
            '레포트': [
                # 한국어 키워드
                '서론', '본론', '결론', '연구', '분석', '조사', '참고문헌', '인용', '논의', '평가', '보고서',
                # 영어 키워드
                'introduction', 'body', 'conclusion', 'research', 'analysis', 'investigation', 
                'references', 'bibliography', 'citation', 'discussion', 'evaluation', 'report',
                'methodology', 'finding', 'result', 'abstract', 'hypothesis', 'data'
            ],
            '기타': []  # 다른 카테고리에 해당하지 않는 경우
        }
        
        # 기계학습 모델 초기화 (선택적으로 사용)
        self.model = None
        
    def count_word_frequency(self, text):
        """문서에서 단어 빈도 계산"""
        words = re.findall(r'\b\w+\b', text.lower())
        return {word: words.count(word) for word in set(words)}
    
    def extract_text_from_pdf(self, pdf_path):
        """PDF 파일에서 텍스트 추출 (일반 + OCR)"""
        try:
            # 기본 PyPDF2를 사용한 텍스트 추출 시도
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # None 체크
                    text += page_text + "\n"
            
            # 추출된 텍스트가 충분하면 반환
            if text and len(text.strip()) > 100:
                return text
            
            # 텍스트가 충분하지 않으면 OCR 시도
            print(f"'{pdf_path}' 파일을 OCR로 처리합니다...")
            return self.extract_text_with_ocr(pdf_path)
            
        except Exception as e:
            print(f"PDF 파일 '{pdf_path}' 일반 추출 오류: {e}")
            # 일반 추출에 실패해도 OCR 시도
            print(f"'{pdf_path}' 파일을 OCR로 처리합니다...")
            return self.extract_text_with_ocr(pdf_path)
    
    def extract_text_with_ocr(self, pdf_path):
        """OCR을 사용한 텍스트 추출"""
        try:
            # pytesseract와 pdf2image를 사용한 OCR 처리
            from pdf2image import convert_from_path
            import pytesseract
            
            # PDF를 이미지로 변환
            images = convert_from_path(pdf_path)
            
            # 각 이미지에서 OCR로 텍스트 추출
            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='kor+eng')  # 한국어 + 영어 지원
                if page_text:
                    text += f"=== Page {i+1} ===\n{page_text}\n\n"
            
            return text
        except Exception as e:
            print(f"OCR 처리 오류: {e}")
            return ""
    
    def train_model(self, training_data):
        """텍스트 분류 모델 학습 (예시 데이터 필요)"""
        # training_data는 {'text': [...], 'label': [...]} 형태의 딕셔너리
        df = pd.DataFrame(training_data)
        
        # TF-IDF와 나이브 베이즈 분류기를 파이프라인으로 구성
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(df['text'], df['label'])
    
    def rule_based_classification(self, text):
        """규칙 기반 문서 분류"""
        scores = {doc_type: 0 for doc_type in self.document_types}
        
        # 각 문서 유형별 키워드 점수 계산
        for doc_type, keywords in self.keywords.items():
            for keyword in keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    scores[doc_type] += 1
        
        # 추가적인 규칙: 문서 구조 분석
        # 자기소개서 패턴 (한국어 + 영어)
        if re.search(r'(저는|제가).{0,50}(지원|희망)', text) or re.search(r'(I am|I\'m|I have).{0,50}(applying|interested in)', text, re.IGNORECASE):
            scores['자기소개서'] += 3
            
        # 이력서 패턴 (한국어 + 영어)
        if re.search(r'(생년월일|연락처|이메일|Date of Birth|DOB|Phone|Contact|Email).{0,30}:', text, re.IGNORECASE) or \
           re.search(r'\d{4}\s*[년~-]\s*\d{1,2}\s*[월~-]', text) or \
           re.search(r'(education|experience|skills|qualification)[\s\n]*:', text, re.IGNORECASE):
            scores['이력서'] += 3
            
        # 학습자료 패턴 (한국어 + 영어)
        if re.search(r'(chapter|단원|목차|학습목표|contents|table of contents|learning objectives|lesson)', text, re.IGNORECASE):
            scores['학습자료'] += 3
            
        # 레포트 패턴 (한국어 + 영어)
        if re.search(r'(참고문헌|reference|인용|출처|bibliography|cited|source|abstract)', text, re.IGNORECASE) or \
           re.search(r'(introduction|methodology|results|conclusion|discussion)', text, re.IGNORECASE):
            scores['레포트'] += 3
        
        # 가장 높은 점수의 문서 유형 반환 (동점일 경우 우선순위 적용)
        max_score = max(scores.values())
        if max_score == 0:
            return '기타'  # 모든 점수가 0이면 기타로 분류
        
        for doc_type in self.document_types:
            if scores[doc_type] == max_score:
                return doc_type
    
    def model_based_classification(self, text):
        """모델 기반 문서 분류"""
        if self.model is None:
            return None  # 모델이 학습되지 않은 경우
        
        prediction = self.model.predict([text])[0]
        return prediction
    
    def classify_pdf(self, pdf_path, use_model=False):
        """PDF 파일 분류하기"""
        text = self.extract_text_from_pdf(pdf_path)
        
        # 텍스트가 여전히 너무 적은 경우 AI 내용 분석 시도
        if not text or len(text.strip()) < 50:
            print(f"'{pdf_path}' 파일을 AI 모델을 사용하여 분석합니다...")
            doc_type = self.classify_with_ai(pdf_path)
            if doc_type:
                return doc_type
            return '분류 불가'
        
        if use_model and self.model is not None:
            return self.model_based_classification(text)
        else:
            # 규칙 기반 분류 실행
            doc_type = self.rule_based_classification(text)
            
            # '기타' 카테고리로 분류된 경우 추가 검사
            if doc_type == '기타':
                # 각 카테고리 키워드 히트 수 확인
                keyword_hits = {
                    category: sum(1 for keyword in keywords if keyword.lower() in text.lower())
                    for category, keywords in self.keywords.items() if category != '기타'
                }
                
                # 키워드 히트가 전혀 없으면 AI 기반 분석 시도
                if sum(keyword_hits.values()) == 0:
                    print(f"'{pdf_path}' 파일을 AI 모델을 사용하여 추가 분석합니다...")
                    ai_doc_type = self.classify_with_ai(pdf_path)
                    if ai_doc_type:
                        return ai_doc_type
                    
                    # AI 분석도 실패하면 '분류 불가'로 변경
                    return '분류 불가'
            
            return doc_type
            
    def classify_with_ai(self, pdf_path):
        """AI 모델을 사용한 PDF 문서 분류"""
        try:
            # 가능하면 텍스트 추출 다시 시도 (OCR 포함)
            text = self.extract_text_from_pdf(pdf_path)
            if not text or len(text.strip()) < 50:
                # 텍스트 추출 실패 시 파일 직접 분석 시도
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                
                # 문서 메타데이터 분석
                metadata = doc.metadata
                
                # 파일명에서 힌트 얻기
                filename = os.path.basename(pdf_path)
                
                # 이미지만 있는 PDF인지 확인
                has_images = False
                has_text = False
                
                for page in doc:
                    if page.get_text().strip():
                        has_text = True
                    if page.get_images():
                        has_images = True
                
                # 문서 특성으로 유형 추정
                if '이력서' in filename or '자기소개서' in filename:
                    return '이력서' if '이력서' in filename else '자기소개서'
                
                if metadata.get('title') and any(kw in metadata.get('title', '').lower() for kw in ['report', '보고서', 'analysis', '분석']):
                    return '레포트'
                
                if metadata.get('title') and any(kw in metadata.get('title', '').lower() for kw in ['교재', '학습', '강의', 'textbook', 'lecture']):
                    return '학습자료'
                
                # 외부 AI 서비스 호출 (OpenAI, Google Cloud Vision 등)
                # 여기서는 구현 예시만 제공, 실제 구현 시 적절한 API 키가 필요합니다
                """
                import openai
                
                # API 키 설정
                openai.api_key = "YOUR_API_KEY"
                
                # 문서 특성 설명
                document_features = f"파일명: {filename}, 텍스트 존재: {has_text}, 이미지 존재: {has_images}"
                if text:
                    # 텍스트가 있으면 일부 추가 (너무 길면 잘라냄)
                    document_features += f", 텍스트 샘플: {text[:500]}"
                
                # AI에 문서 유형 질문
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "PDF 문서의 유형을 분류해주세요. 가능한 유형은 '자기소개서', '이력서', '학습자료', '레포트', '기타'입니다."},
                        {"role": "user", "content": f"다음 문서 특성을 바탕으로 문서 유형을 분류해주세요: {document_features}"}
                    ]
                )
                
                # AI 응답 처리
                ai_response = response.choices[0].message.content.strip().lower()
                
                for doc_type in self.document_types:
                    if doc_type.lower() in ai_response:
                        return doc_type
                """
                
                # 실제 API 호출 없이 문서 특성 기반 추정 반환
                if not has_text and has_images:
                    # 이미지만 있는 경우 학습자료일 가능성 높음
                    return '학습자료'
                
            else:
                # 텍스트 있는 경우 내용 기반 추정
                text_lower = text.lower()
                
                # 각 문서 유형별 특성 점수 계산
                scores = {}
                
                for doc_type, keywords in self.keywords.items():
                    if doc_type == '기타':
                        continue
                    
                    # 키워드 등장 횟수로 점수 계산
                    score = sum(10 for kw in keywords if kw.lower() in text_lower)
                    
                    # 추가 패턴 검사
                    if doc_type == '자기소개서' and (
                        '저는' in text or '제가' in text or 'I am' in text_lower or 'my name is' in text_lower
                    ):
                        score += 20
                    
                    elif doc_type == '이력서' and (
                        re.search(r'\d{4}[-~년]\s*\d{1,2}[-~월]', text) or  # 날짜 패턴
                        re.search(r'(경력|학력|자격증|기술|experience|education|skills)', text_lower)
                    ):
                        score += 20
                    
                    elif doc_type == '학습자료' and (
                        re.search(r'(chapter|단원|목차|학습목표|예제|문제)', text_lower) or
                        text.count('\n\n') > 10  # 많은 문단 구분
                    ):
                        score += 20
                    
                    elif doc_type == '레포트' and (
                        re.search(r'(서론|본론|결론|참고문헌|introduction|conclusion|references)', text_lower) or
                        re.search(r'\[\d+\]', text)  # 인용 패턴
                    ):
                        score += 20
                    
                    scores[doc_type] = score
                
                # 가장 높은 점수 유형 반환
                if scores:
                    max_score = max(scores.values())
                    if max_score > 0:
                        for doc_type, score in scores.items():
                            if score == max_score:
                                return doc_type
            
            return None  # 분류 불가
            
        except Exception as e:
            print(f"AI 기반 분류 오류: {e}")
            return None
    
    def classify_directory(self, directory_path, use_model=False):
        """디렉토리 내 모든 PDF 파일 분류"""
        results = []
        unclassifiable_files = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                doc_type = self.classify_pdf(pdf_path, use_model)
                
                if doc_type == '분류 불가':
                    unclassifiable_files.append({
                        'filename': filename,
                        'path': pdf_path,
                        'reason': '텍스트 추출 실패 또는 충분한 특징 없음'
                    })
                
                results.append({
                    'filename': filename,
                    'path': pdf_path,
                    'type': doc_type
                })
        
        if unclassifiable_files:
            print("\n===== 분류 불가 파일 목록 =====")
            for file in unclassifiable_files:
                print(f"파일명: {file['filename']}")
                print(f"경로: {file['path']}")
                print(f"사유: {file['reason']}")
                print("-" * 50)
        
        return results, unclassifiable_files
    
    def generate_report(self, classification_results, unclassifiable_files=None, output_path=None):
        """분류 결과 보고서 생성"""
        df = pd.DataFrame(classification_results)
        
        # 결과 요약
        summary = pd.DataFrame(df['type'].value_counts()).reset_index()
        summary.columns = ['문서 유형', '개수']
        
        print("\n===== 문서 분류 결과 =====")
        print(f"총 문서 수: {len(df)}")
        print("\n문서 유형별 개수:")
        print(summary)
        
        if unclassifiable_files and len(unclassifiable_files) > 0:
            print(f"\n분류 불가 파일 수: {len(unclassifiable_files)}")
            print("\n===== 분류 불가 파일 상세 정보 =====")
            for i, file in enumerate(unclassifiable_files, 1):
                print(f"{i}. {file['filename']}")
                print(f"   경로: {file['path']}")
                print(f"   사유: {file['reason']}")
        
        if output_path:
            # 기본 분류 결과 저장
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n상세 결과가 '{output_path}'에 저장되었습니다.")
            
            # 분류 불가 파일 정보 저장
            if unclassifiable_files and len(unclassifiable_files) > 0:
                unclassifiable_df = pd.DataFrame(unclassifiable_files)
                unclassifiable_output = output_path.replace('.csv', '_unclassifiable.csv')
                unclassifiable_df.to_csv(unclassifiable_output, index=False, encoding='utf-8-sig')
                print(f"분류 불가 파일 정보가 '{unclassifiable_output}'에 저장되었습니다.")
        
        return df
    

# 사용 예시
if __name__ == "__main__":
    # 필요 라이브러리 확인
    has_all_libraries = check_required_libraries()
    
    classifier = PDFClassifier()
    
    # 단일 PDF 파일 분류
    # pdf_path = "example.pdf"
    # doc_type = classifier.classify_pdf(pdf_path)
    # print(f"파일 '{pdf_path}'의 문서 유형: {doc_type}")
    
    # 디렉토리 내 모든 PDF 파일 분류
    # 경로를 지정할 때는 다음 방법 중 하나를 사용하세요:
    # 1. 원시 문자열(raw string)을 사용 (r 접두사)
    directory_path = r"C:\Users\Lenovo\Desktop\data"  
    # 또는
    # 2. 이중 백슬래시 사용
    # directory_path = "C:\\Users\\Lenovo\\Desktop\\data"
    # 또는 
    # 3. 정방향 슬래시 사용
    # directory_path = "C:/Users/Lenovo/Desktop/data"
    
    if os.path.exists(directory_path):
        print(f"\n{directory_path} 디렉토리의 PDF 파일 분류를 시작합니다...")
        print("이미지 기반 PDF는 OCR 및 AI 분석이 적용됩니다.")
        
        results, unclassifiable_files = classifier.classify_directory(directory_path)
        classifier.generate_report(results, unclassifiable_files, "classification_results.csv")
        
        if unclassifiable_files:
            print(f"\n총 {len(unclassifiable_files)}개 파일을 분류할 수 없었습니다.")
            print("자세한 내용은 'classification_results_unclassifiable.csv' 파일을 확인하세요.")
    else:
        print(f"'{directory_path}' 디렉토리가 존재하지 않습니다.")

    # 모델 학습 예시 (실제 데이터 필요)
    """
    # 예시 학습 데이터
    training_data = {
        'text': [
            # 자기소개서 예시들...
            "저는 귀사에 지원하게 되어 매우 기쁩니다. 제 성장과정과 역량을 말씀드리겠습니다...",
            # 이력서 예시들...
            "이름: 홍길동\n생년월일: 1990.01.01\n학력: 서울대학교\n경력: ABC회사 2018-2022",
            # 학습자료 예시들...
            "Chapter 1. 기초 개념\n이 교재는 학습자들이 쉽게 이해할 수 있도록 구성되었습니다...",
            # 레포트 예시들...
            "서론\n본 연구에서는 다음과 같은 가설을 세웠다...\n본론\n...\n결론\n...\n참고문헌",
        ],
        'label': [
            '자기소개서', '이력서', '학습자료', '레포트'
            # ... 더 많은 예시들 ...
        ]
    }
    classifier.train_model(training_data)
    
    # 학습된 모델로 분류
    results = classifier.classify_directory(directory_path, use_model=True)
    classifier.generate_report(results, "model_classification_results.csv")
    """