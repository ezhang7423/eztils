import typer
from typing_extensions import Annotated
from eztils import json

def dataclass_option(dataclass):
    def parser(x):
        if isinstance(x, dataclass):
            return x
        if isinstance(x, str):            
            try:
                x = json.loads(x)
            except json.decoder.JSONDecodeError:
                raise typer.BadParameter(
                    f"Could not parse {x} as a JSON string."
                )
            return dataclass(**x)
        if isinstance(x, dict):
            return dataclass(**x)
    
    return Annotated[ # create new type
        dataclass, typer.Option(parser=parser)
    ]

