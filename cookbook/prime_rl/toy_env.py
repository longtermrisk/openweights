import verifiers as vf
from datasets import Dataset


def load_environment(num_examples: int = 16) -> vf.Environment:
    rows = []
    for i in range(num_examples):
        rows.append(
            {
                "question": f"Repeat the number: {i}",
                "answer": str(i),
                "info": {},
                "task": "toy-env",
            }
        )

    dataset = Dataset.from_list(rows)
    parser = vf.Parser()

    def exact_match_reward(completion, answer, **kwargs) -> float:
        response = (parser.parse_answer(completion) or "").strip()
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[exact_match_reward])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="Return only the number you were asked to repeat.",
        parser=parser,
        rubric=rubric,
    )
