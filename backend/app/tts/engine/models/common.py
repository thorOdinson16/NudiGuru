from pydantic import BaseModel


class Language(BaseModel):
    sourceLanguage: str

    # @validator('sourceLanguage', pre=True)
    # def blank_string_in_language(cls, value, field):
    #     if value == "":
    #         raise ValueError('sourceLanguage cannot be empty')
    #     return value
