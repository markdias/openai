# Footballer Info CLI

This repository contains a small Python command-line tool that sends your
queries about footballers (soccer players) to OpenAI's GPT models. It is useful
for gathering quick scouting reports, background information, and notable
statistics for players you're interested in. Responses now wrap up with two
follow-up questions about the subject and highlight a teammate who has played
with them.

## Getting started

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies (which include [pandas](https://pandas.pydata.org/) for clean table rendering):

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

### Interactive question menu

If you're not sure what to ask, you can launch an interactive menu of sample
queries. The tool will list several ready-made scouting prompts and send the
selected one to the API:

```bash
python footballer_app.py --menu
```

The tool prints the model's response directly to standard output. When a
career-history table is present, it is re-rendered with pandas so the columns
line up neatly in your terminal.

### Career-history tables

If you simply pass a player's name (for example `"Sophia Smith"`), the tool will
automatically request their club and international career history formatted as a
chronological Markdown table. You can explicitly control this behaviour:

```bash
# Force the table output regardless of the query phrasing
python footballer_app.py "Lionel Messi scouting report" --career-table

# Opt out of the automatic table when you just want narrative analysis
python footballer_app.py "Ada Hegerberg" --no-career-table
```

## Sending results to Slack (#general in markdias workspace)

You can optionally forward the generated scouting report to your Slack workspace.
Create a *bot* token in the **markdias** workspace with the `chat:write`
permission (its value should begin with `xoxb-`) and export it as an
environment variable:

```bash
export SLACK_BOT_TOKEN="xoxb-your-token"
```

When you run the CLI, supply the target channel (for example `#general`) and the
tool will post the message after printing it locally:

```bash
python footballer_app.py "Update me on Trinity Rodman's recent performances" \
  --slack-channel "#general"
```

Alternatively, you can provide the token inline with `--slack-token` if you do
not want to rely on an environment variable.
