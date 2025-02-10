# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김영만
- 리뷰어 : 이정우


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        ![image](https://github.com/user-attachments/assets/7f2cd106-fe0c-4b20-a1dc-5c500c320ad2)
    어텐션을 이용하여 완료하셧습니다.

Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다. 분석단계, 정제단계, 정규화와 불용어 제거, 데이터셋 분리, 인코딩 과정이 빠짐없이 체계적으로 진행되었다.
너무 많으므로 생략하겠습니다.

![image](https://github.com/user-attachments/assets/be8760dd-9cec-4a13-83bb-7582df8001ba)

모델 학습이 진행되면서 train loss와 validation loss가 감소하는 경향을 그래프를 통해 확인했으며, 실제 요약문에 있는 핵심 단어들이 요약 문장 안에 포함되었다.


원문 : Weeks after ex-CBI Director Alok Verma told the Department of Personnel and Training to consider him retired, the Home Ministry asked him to join work on the last day of his fixed tenure as Director on Thursday. The ministry directed him to immediately join as DG, Fire Services, the post he was transferred to after his removal as CBI chief.
실제 요약 : Govt directs Alok Verma to join work 1 day before his retirement
예측 요약 : 


원문 : Andhra Pradesh CM N Chandrababu Naidu has said, "When I met then US President Bill Clinton, I addressed him as Mr Clinton, not as 'sir'. (PM Narendra) Modi is my junior in politics...I addressed him as sir 10 times." "I did this...to satisfy his ego in the hope that he will do justice to the state," he added.
실제 요약 : Called PM Modi 'sir' 10 times to satisfy his ego: Andhra CM
예측 요약 : 


원문 : Congress candidate Shafia Zubair won the Ramgarh Assembly seat in Rajasthan, by defeating BJP's Sukhwant Singh with a margin of 12,228 votes in the bypoll. With this victory, Congress has taken its total to 100 seats in the 200-member assembly. The election to the Ramgarh seat was delayed due to the death of sitting MLA and BSP candidate Laxman Singh.
실제 요약 : Cong wins Ramgarh bypoll in Rajasthan, takes total to 100 seats
예측 요약 : Congress candidate Shafia Zubair won the Ramgarh Assembly seat in Rajasthan, by defeating BJP's Sukhwant Singh with a margin of 12,228 votes in the bypoll.


두 요약 결과를 문법완성도 측면과 핵심단어 포함 측면으로 나누어 비교하고 분석 결과를 표로 정리하여 제시하였다.

    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

![image](https://github.com/user-attachments/assets/905c6f8c-3dea-4f0c-91c8-097abaab3828)

네. 주석이나 이런부분이 잘되어있습니다.

        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
     
뉴스 요약에 대한 프로젝트를 진행 해 보았는데, lecture 이상으로 진행 하지 못하여 아쉬움이 있었지만 에테션에 대한 이해를 높일 수 있었어 좋았습니다.

그리고, 추상적인 요약에서 예측 부분은 shape 문제로 완료 하지 못하였습니다.

어텐션의 좋은 예제로 계속 업데이트 해 나갈 생각입니다.

로 회고록을 작성하여 오류부분을 남겨주셨습니다.
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

![image](https://github.com/user-attachments/assets/08de783f-cb09-456e-8127-c8603e875353)

위와 같습니다.
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
![image](https://github.com/user-attachments/assets/f8e59a4e-eb16-4d1a-b95b-0ffb180067c6)


함수를 잘만들어서 이해하기 쉬웠습니다.


# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

부족한부분과 하신 부분이 명확하게 설명되어 있어서 이해하기도 좋았고 배울점도 많았습니다.
```

