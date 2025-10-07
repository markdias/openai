# Footballer Info CLI

This repository contains a small Python command-line tool that sends your
queries about footballers (soccer players) to OpenAI's GPT models. It is useful
for gathering quick scouting reports, background information, and notable
statistics for players you're interested in.

## Getting started

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key in the environment:

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

## Usage

Run the script with a football-related query:

```bash
python footballer_app.py "Tell me about the career of Alexia Putellas"
```

Optional flags let you customize the model and sampling temperature:

```bash
python footballer_app.py "How is Bukayo Saka performing this season?" --model gpt-4.1 --temperature 0.4
```

The tool prints the model's response directly to standard output.
