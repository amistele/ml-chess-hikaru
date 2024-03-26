import pandas as pd
hikaru_solutions = pd.read_csv('TRAINING_SOLUTIONS_hikaru_1239497.csv')
hikaru_solutions_10k = hikaru_solutions.head(10000)
hikaru_solutions_10k.to_csv('TRAINING_SOLUTIONS_hikaru_10k.csv',index=False)
