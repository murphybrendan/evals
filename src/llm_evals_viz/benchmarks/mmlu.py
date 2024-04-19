from dataclasses import dataclass


@dataclass
class MMLU:
    path: str = "lighteval/mmlu"
    name: str = "abstract_algebra"
    split: str = "test"

    def build_prompt(self, question: str, choices: list[str]):
        numbered_choices = "\n".join([f"{number}. {choice}" for number, choice in zip(range(len(choices)), choices)])
        prompt = f"""You will be given a multiple choice question and four possible answers. Your job is to provide the correct answer. Return just the number of the correct answer and nothing else.
            Here is the question:
            {question}

            Here are the possible answers:
            {numbered_choices}
        """
        return prompt
    
    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path} name={self.name} split={self.split})"
