from typing import List, Optional
import pandas as pd
import re

'''
    Old, other preprocessor.
    The one used is in preprocess.py .
    Example of well-written preprocessor class.
'''

class Preprocessor:
    @classmethod
    def normalize_name(cls, name: str) -> str:
        return " ".join([word.strip(",()[].\"'") for word in name.split(" ")])

    @classmethod
    def ticket_split(cls, ticket: str) -> (Optional[str], Optional[int]):
        ticket_item, ticket_number = None, None

        ticket = ticket.split(" ")
        if len(ticket) == 2:
            ticket_item = ticket[0]
            ticket_number = int(ticket[1])
        elif len(ticket) == 1:
            # If the element starts with a letter, is an item. Else, a number.
            if re.match(r'^[a-zA-z]', ticket[0]):
                ticket_item = ticket[0]
            else:
                ticket_number = int(ticket[0])
        return ticket_item, ticket_number

    @classmethod
    def transform_dummy_variables(cls, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df[features])

    @classmethod
    def preprocess(cls, df):
        df = df.copy()
        df["Name"] = df["Name"].apply(cls.normalize_name)
        df[['Ticket_item', 'Ticket_number']] = df['Ticket'].apply(
            lambda ticket: pd.Series(cls.ticket_split(ticket))
        )
        df['Ticket_item'] = df['Ticket_item'].astype('object')
        df['Ticket_number'] = df['Ticket_number'].astype('int64')
        return df


def preprocess(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    def ticket_number(x):
        return x.split(" ")[-1]

    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df
