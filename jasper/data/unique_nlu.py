import pandas as pd


def compute_pnr_name_city():
    data = pd.read_csv("./customer_utterance_processing/customer_provide_answer.csv")

    def unique_pnr_count():
        pnr_data = data[data["Input.Answer"] == "ZZZZZZ"]
        unique_pnr_set = {
            t
            for n in range(1, 5)
            for t in pnr_data[f"Answer.utterance-{n}"].tolist()
            if "ZZZZZZ" in t
        }
        return len(unique_pnr_set)

    def unique_name_count():
        pnr_data = data[data["Input.Answer"] == "John Doe"]
        unique_pnr_set = {
            t
            for n in range(1, 5)
            for t in pnr_data[f"Answer.utterance-{n}"].tolist()
            if "John Doe" in t
        }
        return len(unique_pnr_set)

    def unique_city_count():
        pnr_data = data[data["Input.Answer"] == "Heathrow Airport"]
        unique_pnr_set = {
            t
            for n in range(1, 5)
            for t in pnr_data[f"Answer.utterance-{n}"].tolist()
            if "Heathrow Airport" in t
        }
        return len(unique_pnr_set)

    def unique_entity_count(entity_template_tags):
        # entity_data = data[data['Input.Prompt'] == entity_template_tag]
        entity_data = data
        unique_entity_set = {
            t
            for n in range(1, 5)
            for t in entity_data[f"Answer.utterance-{n}"].tolist()
            if any(et in t for et in entity_template_tags)
        }
        return len(unique_entity_set)

    print('PNR', unique_pnr_count())
    print('Name', unique_name_count())
    print('City', unique_city_count())
    print('Payment', unique_entity_count(['KPay', 'ZPay', 'Credit Card']))


def compute_date():
    entity_template_tags = ['27 january', 'December 18']
    data = pd.read_csv("./customer_utterance_processing/customer_provide_departure.csv")
    # data.sample(10)

    def unique_entity_count(entity_template_tags):
        # entity_data = data[data['Input.Prompt'] == entity_template_tag]
        entity_data = data
        unique_entity_set = {
            t
            for n in range(1, 5)
            for t in entity_data[f"Answer.utterance-{n}"].tolist()
            if any(et in t for et in entity_template_tags)
        }
        return len(unique_entity_set)

    print('Date', unique_entity_count(entity_template_tags))


def compute_option():
    entity_template_tag = 'third'
    data = pd.read_csv("./customer_utterance_processing/customer_provide_flight_selection.csv")

    def unique_entity_count():
        entity_data = data[data['Input.Prompt'] == entity_template_tag]
        unique_entity_set = {
            t
            for n in range(1, 5)
            for t in entity_data[f"Answer.utterance-{n}"].tolist()
            if entity_template_tag in t
        }
        return len(unique_entity_set)

    print('Option', unique_entity_count())


compute_pnr_name_city()
compute_date()
compute_option()
