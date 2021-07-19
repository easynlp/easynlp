import easynlp


def test_single_summarization():
    data = {
        "text": [
            """The warning begins at 22:00 GMT on Saturday and
               ends at 10:00 on Sunday. The ice could lead to
               difficult driving conditions on untreated roads
               and slippery conditions on pavements, the weather
               service warned. Only the southernmost counties and
               parts of the most westerly counties are expected
               to escape. Counties expected to be affected are
               Carmarthenshire, Powys, Ceredigion, Pembrokeshire,
               Denbighshire, Gwynedd, Wrexham, Conwy, Flintshire,
               Anglesey, Monmouthshire, Blaenau Gwent,
               Caerphilly, Merthyr Tydfil, Neath Port Talbot,
               Rhondda Cynon Taff and Torfaen.""",
        ]
    }
    input_column = "text"
    output_column = "summarization"
    output_dataset = easynlp.summarization(data, input_column, output_column)
    summarized_text = [
        'The Met Office has issued a yellow "be aware" warning for ice across much of Wales.',
    ]
    assert len(output_dataset) == 1
    assert output_dataset[output_column] == summarized_text


def test_summarization():
    data = {
        "text": [
            """The warning begins at 22:00 GMT on Saturday and
               ends at 10:00 on Sunday. The ice could lead to
               difficult driving conditions on untreated roads
               and slippery conditions on pavements, the weather
               service warned. Only the southernmost counties and
               parts of the most westerly counties are expected
               to escape. Counties expected to be affected are
               Carmarthenshire, Powys, Ceredigion, Pembrokeshire,
               Denbighshire, Gwynedd, Wrexham, Conwy, Flintshire,
               Anglesey, Monmouthshire, Blaenau Gwent,
               Caerphilly, Merthyr Tydfil, Neath Port Talbot,
               Rhondda Cynon Taff and Torfaen.""",
            """The warning begins at 22:00 GMT on Saturday and
               ends at 10:00 on Sunday. The ice could lead to
               difficult driving conditions on untreated roads
               and slippery conditions on pavements, the weather
               service warned. Only the southernmost counties and
               parts of the most westerly counties are expected
               to escape. Counties expected to be affected are
               Carmarthenshire, Powys, Ceredigion, Pembrokeshire,
               Denbighshire, Gwynedd, Wrexham, Conwy, Flintshire,
               Anglesey, Monmouthshire, Blaenau Gwent,
               Caerphilly, Merthyr Tydfil, Neath Port Talbot,
               Rhondda Cynon Taff and Torfaen""",
        ]
    }
    input_column = "text"
    output_column = "summarization"
    output_dataset = easynlp.summarization(data, input_column, output_column)
    summarized_text = [
        'The Met Office has issued a yellow "be aware" warning for ice across much of Wales.',
        'The Met Office has issued a yellow "be aware" warning for ice across most of Wales.',
    ]
    assert len(output_dataset) == 2
    assert output_dataset[output_column] == summarized_text
