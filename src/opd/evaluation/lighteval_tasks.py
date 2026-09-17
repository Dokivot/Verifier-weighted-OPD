from dataclasses import replace

from lighteval.tasks.default_tasks import aime24, math_500

MAX_GENERATION_TOKENS = 4096

TASKS_TABLE = [
    replace(math_500, generation_size=MAX_GENERATION_TOKENS, version=3),
    replace(aime24, generation_size=MAX_GENERATION_TOKENS, version=3),
]
