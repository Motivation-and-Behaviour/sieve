from openai import OpenAI
from openai.types import Batch, FileObject, FilePurpose
from openai.types.responses.response_input_param import ResponseInputItemParam

from sieve.config import (
    ABSTRACT_SCREENING_INSTRUCTIONS,
    EXCLUSION_CRITERIA,
    FULLTEXT_SCREENING_INSTRUCTIONS,
    INCLUSION_CRITERIA,
    INCLUSION_HEADER,
    STUDY_OBJECTIVES,
)
from sieve.credentials import load_token
from sieve.datamodels import ScreeningDecision


class OpenAIManager:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5.2"):
        if api_key is None:
            api_key = load_token("OPENAI_TOKEN")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def screen_record_abstract(self, abstract: str):
        inputs = self._build_abstract_prompt(abstract)

        response = self.client.responses.parse(
            model=self.model,
            input=inputs,
            text_format=ScreeningDecision,
        )
        return response.output_parsed

    def screen_record_fulltext(self, pdf_path: str):
        file = self.upload_file(pdf_path)

        inputs = self._build_fulltext_prompt(file.id)

        response = self.client.responses.parse(
            model=self.model,
            input=inputs,
            text_format=ScreeningDecision,
        )

        return response.output_parsed

    def prepare_abstract_body(self, abstract: str) -> dict:
        """Returns the body for the batch request (messages + schema)."""
        inputs = self._build_abstract_prompt(abstract)
        return self._build_structured_payload(inputs, ScreeningDecision)

    def prepare_fulltext_body(self, file_id: str) -> dict:
        """
        Requires a file_id.
        The IntegrationManager must call upload_file first.
        """
        inputs = self._build_fulltext_prompt(file_id)
        return self._build_structured_payload(inputs, ScreeningDecision)

    def parse_screening_decision(self, json_content: str) -> ScreeningDecision:
        return ScreeningDecision.model_validate_json(json_content)

    def upload_file(
        self, file_path: str, purpose: FilePurpose = "user_data"
    ) -> FileObject:
        with open(file_path, "rb") as f:
            return self.client.files.create(file=f, purpose=purpose)

    def create_batch(self, filename: str, batch_type: str) -> Batch:
        batch_input_file = self.upload_file(filename, purpose="batch")
        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": batch_type},
        )

    def retrieve_batch(self, batch_id: str) -> Batch:
        return self.client.batches.retrieve(batch_id)

    def create_batch_row(
        self, custom_id: str, body: dict, url: str = "/v1/responses"
    ) -> dict:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": url,
            "body": body,
        }

    def _build_structured_payload(self, input: list, pydantic_model) -> dict:
        return {
            "model": self.model,
            "input": input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": pydantic_model.__name__,
                    "schema": pydantic_model.model_json_schema(),
                    "strict": True,
                }
            },
        }

    def _build_abstract_prompt(self, abstract: str) -> list[ResponseInputItemParam]:
        prompt = (
            STUDY_OBJECTIVES
            + "\n"
            + INCLUSION_HEADER
            + "\n"
            + "Inclusion criteria (all must be met):\n"
            + self._number_criteria(INCLUSION_CRITERIA)
            + "\n"
            + "Exclusion criteria (any triggers exclusion):\n"
            + self._number_criteria(EXCLUSION_CRITERIA)
            + "\n"
            + ABSTRACT_SCREENING_INSTRUCTIONS
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Abstract:\n{abstract}"},
        ]

    def _build_fulltext_prompt(self, file_id: str) -> list[ResponseInputItemParam]:
        prompt = (
            STUDY_OBJECTIVES
            + "\n"
            + INCLUSION_HEADER
            + "\n"
            + "Inclusion criteria (all must be met):\n"
            + self._number_criteria(INCLUSION_CRITERIA)
            + "\n"
            + "Exclusion criteria (any triggers exclusion):\n"
            + self._number_criteria(EXCLUSION_CRITERIA)
            + "\n"
            + FULLTEXT_SCREENING_INSTRUCTIONS
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "input_file", "file_id": file_id}]},
            {"role": "system", "content": prompt},
        ]

    @staticmethod
    def _number_criteria(criteria: list[str]) -> str:
        return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
