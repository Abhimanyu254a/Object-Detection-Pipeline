from transformers import pipeline

class SummarizationModel:
    def __init__(self, model_name='sshleifer/distilbart-cnn-12-6'):
        """
        Initializes the summarization model using Hugging Face's transformers pipeline.
        The default model used is 'distilbart-cnn-12-6', which is fine-tuned for summarization.
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_words=100) -> str:
        """
        Summarize the given text using the pre-trained model.
        Args:
            - text (str): Input text to be summarized.
            - max_words (int): Maximum number of words in the summary.

        Returns:
            - summary (str): The summarized text.
        """
        if not text.strip():
            return "No valid text for summarization."

        word_count = len(text.split())

        # Skip summarization for very short text
        if word_count <= 5:
            return text

        # Define the maximum and minimum length for the summary
        max_length = min(max_words, word_count)
        min_length = min(max(10, max_length // 2), max_length - 1)

        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary.strip()  # Return a clean summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"
