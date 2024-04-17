import time

from anthropic import Anthropic, RateLimitError, InternalServerError

from evals.models import Model

client = Anthropic()

class Claude3Model(Model):

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"

    def get_completion(self, messages, max_retries=5, retry_delay=60):
        retries = 0
        while retries < max_retries:
            try:
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=5,
                    messages=messages
                )
                return response.content[0].text
            except (RateLimitError, InternalServerError) as e:
                print(f"Error: {e}")
                retries += 1
                if retries == max_retries:
                    print("Maximum number of retries reached. Aborting.")
                    return None
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        return None
    
    def get_completion_for_prompt(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.get_completion(messages)