# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김영만
- 리뷰어 : 정상헌


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - vocab_size 변화에 따른 모델별 성능을 산출해주셨습니다.
    - ![Image](https://github.com/user-attachments/assets/92791757-0973-4f21-9ede-7b17e13ba9ca)  
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - tf-idf 행렬을 계산하여 문서를 벡터화하는 과정을 코드를 처음 보는 사람도 이해하기 쉽도록 잘 정리해주셨습니다.  
    - ![Image](https://github.com/user-attachments/assets/5994bd0d-467f-4efc-82d4-6628b807248a)  
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - random forest의 n_estimators를 조절하면서 실험하였습니다.  
    - hard voting을 했을 때의 성능도 측정하셨습니다.
    - ![Image](https://github.com/user-attachments/assets/f163b9e4-404b-40a2-a564-c9df2db13203)  
    - ![Image](https://github.com/user-attachments/assets/f86cd4e0-aa13-4f62-9c52-d78e5a2bf134)  
        
- [X]  **4. 회고를 잘 작성했나요?**
    - ![Image](https://github.com/user-attachments/assets/7d3cd57c-d7d5-465d-8046-810efcfcda98)  
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 정수 인코딩을 텍스트로 변환하는 코드를 Pythonic하게 작성해주셨습니다.  
    - ![Image](https://github.com/user-attachments/assets/d8607e9a-c108-4ef3-9403-fa9528345dac)  


# 회고(참고 링크 및 코드 개선)
```
저는 하지 못했던 랜덤 포레스트의 n_estimator 증가에 따른 성능과 hard voting을 했을 때의 성능도 측정하셔서 인상깊었습니다.
고생 많으셨습니다 :)
```
