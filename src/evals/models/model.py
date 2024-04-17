
class Model:
    """
    Defines and interface for models for the purpose of performance evaluation.
    """

    def get_completion(self, messages):
        """
        Get the completion for the given messages.

        Args:
            messages: List of messages to be completed.

        Returns:
            The completion of the messages.
        """
        raise NotImplementedError()
    

    def get_completion_for_prompt(self, prompt: str):
        """
        Get the completion for the given prompt.
        
        Args:
            prompt: Prompt for the LLM to respond to.

        Returns:
            The response from the LLM.
        """
        raise NotImplementedError()