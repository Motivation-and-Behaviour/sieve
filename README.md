# Sieve

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/Motivation-and-Behaviour/sieve)
[![Actions status](https://github.com/Motivation-and-Behaviour/sieve/workflows/Tests/badge.svg)](https://github.com/Motivation-and-Behaviour/sieve/actions)
[![codecov](https://codecov.io/gh/Motivation-and-Behaviour/sieve/graph/badge.svg?token=avmeh0jcwS)](https://codecov.io/gh/Motivation-and-Behaviour/sieve)

Sieve (Systematic Identification, EValuation, and Extraction) is a set of tools to conduct abstract and full-text screening, and (eventually) data extraction.
It is the generic version of the project-specific tool [Bigger Picker](https://github.com/Motivation-and-Behaviour/bigger_picker).

## Features

The sieve package is still in development and does not yet have all the features of the Bigger Picker, but it includes the following:

- **Abstract screening**: Using OpenAI's models to screen abstracts based on a provided protocol and use this information to vote on Rayyan.
- **Full-text screening**: Using OpenAI's models to screen full texts by taking the full-text PDF from Rayyan.

Features that are available in Bigger Picker but not yet in Sieve include:

- **Data extraction**: Using OpenAI's models to extract data from full texts based on a provided protocol and use this information to populate an Airtable base.

Features in Bigger Picker that are not planned for Sieve include:

- **Asana integration**: Syncing article statuses between Airtable and Asana.

## Prerequisites

### API Keys and Tokens

- **Rayyan API Key**: Used to interact with the Rayyan Review.
   Note that Rayyan only provies API keys for Pro accounts.
   You can either provide the `rayyan_tokens.json` file directly, or store the credentials in the environment variable `RAYYAN_CREDS_JSON` as a JSON string.
- **OpenAI API Key**: Used to access OpenAI's language models for screening and data extraction.

### Environment

The project is built and tested using Python 3.13, but should work on Python 3.11 and above.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Motivation-and-Behaviour/sieve.git
   cd sieve
   ```

2. (Optional) Create and activate a virtual environment:

   ```sh
   python -m venv .venv
   source ./venv/bin/activate #On Windows, use .\venv\Scripts\activate
   ```

3. Install dependencies:

   ```sh
   pip install -e .
   ```

    You can also optionally install the development dependencies:

   ```sh
   pip install -e .[dev, docs]
   ```

4. Set up your `.env` file with the required API keys and configuration values.
   The provided `.env.example` can be used as a template.

## Configuration

Most settings for the project are set in the [sieve/config.py](sieve/config.py) file.
You will need to set:

- `RAYYAN_REVIEW_ID`: The ID of your Rayyan review.
- `RAYYAN_EXCLUSION_LABELS` - these need to match the exact order of `EXCLUSION_CRITERIA`, and they have to start with `__EXR__`
- `STUDY_OBJECTIVES`
- `INCLUSION_CRITERIA`
- `EXCLUSION_CRITERIA`

Use the provided parameters as examples for how to format these settings.

You will also need a `rayyan_tokens.json` file for Rayyan API authentication.

## Usage

Once installed, you can use the command-line interface (CLI) to interact with sieve.

The CLI provides three main commands:

- **Screen fulltexts:**

  ```sh
  sieve screenft
  ```

- **Screen abstracts:**

  ```sh
  sieve screenabstract
  ```

- **Monitor Rayyan for new articles to screen:**

  ```sh
  sieve monitor
  ```

Note that `monitor` runs continuously.
Since it runs continuously, it can take advantage of batch processing from OpenAI, which halves the costs but will take slightly longer.
The `monitor` command will check if a batch is finished and upload the results to Rayyan, and also check if new articles have been added to Rayyan and add them to a batch for screening.
It is the recommended way to use sieve.

Appending `--help` to a command will provide additional options and usage information.

## License

This project is licensed under the MIT License.
