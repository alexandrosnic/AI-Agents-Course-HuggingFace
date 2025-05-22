from smolagents import Tool, LiteLLMModel
from smolagents.cli import load_model
from typing import Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSummarizationTool(Tool):
    """
    A tool for reading and summarizing text, emphasizing its most important points.
    """

    name = "text_summarization_tool"
    description = "Reads and summarizes a given text, highlighting its most important points."
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A natural language description of the task to perform (e.g., 'Summarize the following text: ...'). It may be a document, article, or report.",
            "required": True,
        }
    }
    output_type = "object"

    def __init__(self, model: LiteLLMModel):
        # Initialize the language model
        self.model = model

    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
        """
        Splits the text into smaller chunks using RecursiveCharacterTextSplitter.
        Args:
            text: The text to split.
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The number of overlapping characters between chunks.
        Returns:
            A list of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def summarize_chunk(self, chunk: str) -> str:
        """
        Summarizes a single chunk of text using the language model.
        Args:
            chunk: The text chunk to summarize.
        Returns:
            The summary of the chunk.
        """
        prompt = f"""
        Summarize the following text, emphasizing its most important points:
        {chunk}
        """
        # response = self.model.run(prompt)
        # return response.strip()
        response = self.model.complete(prompt=prompt).strip()  
        return response  


    def forward(self, prompt: str) -> Dict:
        """
        Summarizes the input text by splitting it into chunks and summarizing each chunk.
        Args:
            prompt: A string containing the text to summarize.
        Returns:
            A dictionary containing the final summary.
        """
        if not prompt:
            raise ValueError("The 'prompt' field is required.")

        # Split the text into chunks
        chunks = self.split_text(prompt)

        # Summarize each chunk
        summaries = [self.summarize_chunk(chunk) for chunk in chunks]

        # Combine the summaries into a final summary
        final_summary = " ".join(summaries)

        return {"summary": final_summary}