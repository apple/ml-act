# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from act.evaluations import evaluate_0shot

SYSTEM_PROMPT = [
    "You are a chatbot that tells if a sentence is about fantasy.",
    "You are a chatbot that tells if a sentence is about football.",
]
ANSWERS = ["no", "yes"]
DATA = {
    "sentence": [
        "Hello, my name is John and I play a sport with a ball.",
        "Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "Messi is the best player.",
        "I once saw a strange insect.",
    ],
}

SYSTEM_PROMPT_MC = [
    "Given the following 2 sentences, select the one that talks more about fantasy.",
    "Given the following 2 sentences, select the one that talks more about football.",
]
ANSWERS_MC = ["A", "B"]
DATA_MC = {
    "sentence": [
        "Hello, my name is John and I play a sport with a ball.",
        "Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "Messi is the best player.",
        "I once saw a strange insect.",
    ],
}


@pytest.mark.skip(reason="Uses Llama-3-8B-instruct, too large.")
@pytest.mark.parametrize(
    "system_prompt,answers,data,use_second_csv,prepend_answers",
    [
        (SYSTEM_PROMPT, ANSWERS, DATA, False, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, False, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, True, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, False, 1),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, True, 1),
    ],
)
def test_0shot_e2e(system_prompt, answers, data, use_second_csv, prepend_answers):
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        csv_file = Path(tempfolder) / "test.csv"
        out_file = Path(tempfolder) / "out.csv"
        df = pd.DataFrame(data=data)
        df.to_csv(csv_file)

        second_csv_argv = []
        if use_second_csv:
            csv_file2 = Path(tempfolder) / "test.csv"
            df.to_csv(csv_file2)
            second_csv_argv = [
                "--data-path2",
                str(csv_file2),
            ]

        parser = evaluate_0shot.get_parser()
        args = parser.parse_args(
            [
                "--device",
                "cpu",
                "--system-prompt",
                *system_prompt,
                "--system-answers",
                *answers,
                "--prepend-answers",
                str(prepend_answers),
                "--col-sentence1",
                "sentence",
                "--data-path",
                str(csv_file),
                "--output-file",
                str(out_file),
                *second_csv_argv,
            ]
        )
        evaluate_0shot.main(args)
        df_out = pd.read_csv(out_file)

        assert len(df_out) == 2
        assert "q0_llm_answer" in df_out.columns
        assert "q1_llm_answer" in df_out.columns
        assert df_out["q0_llm_answer"].values[1] in ["yes", "A"]
