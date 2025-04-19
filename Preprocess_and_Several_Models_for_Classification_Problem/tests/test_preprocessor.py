from Preprocess_and_Several_Models_for_Classification_Problem.src.preprocessor import Preprocessor
import pandas as pd

def test_normalize_name():
    name = "Ms. Wayne, Alice"
    prepr_name = Preprocessor.normalize_name(name)
    assert prepr_name == "Ms Wayne Alice"

def test_ticket_only_number():
    ticket = "5000"
    prepr_ticket_item, prepr_ticket_number = Preprocessor.ticket_split(ticket)
    assert prepr_ticket_item is None
    assert prepr_ticket_number == 5000

def test_ticket_only_item():
    ticket = "A/C"
    prepr_ticket_item, prepr_ticket_number = Preprocessor.ticket_split(ticket)
    assert prepr_ticket_item == "A/C"
    assert prepr_ticket_number is None

def test_ticket_item_number():
    ticket = "A/C 5000"
    prepr_ticket_item, prepr_ticket_number = Preprocessor.ticket_split(ticket)
    assert prepr_ticket_item == "A/C"
    assert prepr_ticket_number == 5000

def test_preprocess_name_ticket_number():
    data = {
        'Name':
            ['Ms. Wayne, Alice',
             'Bobby',
             'Charlie "Pepe" Walls'
             ],
        'Ticket': "837"
    }
    df = pd.DataFrame(data)

    preprocessed_df = Preprocessor.preprocess(df)

    test_data = {
        'Name':
            [
                'Ms Wayne Alice',
                'Bobby',
                'Charlie Pepe Walls'
            ],
        'Ticket': 837,
        'Ticket_number': 837,
        'Ticket_item': None
    }
    test_df = pd.DataFrame(test_data)

    assert preprocessed_df['Name'].equals(test_df['Name'])
    assert preprocessed_df['Ticket_number'].equals(test_df['Ticket_number'])
    assert preprocessed_df['Ticket_item'].equals(test_df['Ticket_item'])
