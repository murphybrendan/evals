import lm_eval
import logging


class EvaluationSweep:
    def __init__(self, 
                 models: list = None, 
                 prompting_templates: list = None,
                 n_shots: list = None, 
                 chain_of_thought: bool = False, 
                 tasks: list = None):
        self.models = models or []
        self.prompting_templates = prompting_templates or []
        self.n_shots = n_shots or [None]
        self.chain_of_thought = chain_of_thought
        self.tasks = tasks or []

    def add_model(self, model):
        self.models.append(model)

    def add_prompting_template(self, template):
        self.prompting_templates.append(template)

    def add_n_shot_value(self, n):
        self.n_shots.append(n)

    def add_task(self, task):
        self.tasks.append(task)

    def enable_chain_of_thought(self):
        self.chain_of_thought = True

    def disable_chain_of_thought(self):
        self.chain_of_thought = False

    def get_configurations(self):
        configurations = []
        for model in self.models:
            for n in self.n_shots:
                configuration = {
                    'model': model,
                    'n_shot': n,
                    'chain_of_thought': self.chain_of_thought
                }
                configurations.append(configuration)
        logging.debug(f"Configurations for sweep: {configurations}")
        return configurations
    
    def run(self):
        configurations = self.get_configurations()
        results = []
        task_manager = lm_eval.tasks.TaskManager()
        for configuration in configurations:
            result = lm_eval.simple_evaluate(
                model=configuration['model'][0],
                model_args=configuration['model'][1],
                tasks=self.tasks, 
                num_fewshot=configuration['n_shot'],
                task_manager=task_manager)
            results.append(result)
        return results