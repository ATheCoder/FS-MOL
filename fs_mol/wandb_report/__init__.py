import pandas
import wandb
from dataclasses import asdict


def generate_evaluation_chart(test_results):
    if wandb.run != None:
        test_results_list = list(map(lambda x: asdict(x), test_results))
        
        test_results_df = pandas.DataFrame(test_results_list)
        
        data = test_results_df.groupby('num_train', as_index=False).mean(numeric_only=True)
        
        table_data = [[x, y] for (x, y) in zip(data['num_train'], data['optimistic_delta_auc_pr'])]
        
        table = wandb.Table(data=table_data, columns=['num_train', 'optimistic_delta_auc_pr'])
        
        wandb.log({"Evaluation": wandb.plot.line(table=table, x="num_train", y="optimistic_delta_auc_pr", title="Delta AUCPR for Different Sample Sizes")})