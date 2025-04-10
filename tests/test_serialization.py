import enum

import yaml

from finetuning.utils.serialization import ConfigDumper


def test_config_dumper_dictionary():
    data = {"key": "value"}
    expected_output = "key: value\n"
    assert yaml.dump(data, Dumper=ConfigDumper) == expected_output


def test_config_dumper_enum():
    class MyEnum(str, enum.Enum):
        VALUE = "enum_value"

    data = {"key": MyEnum.VALUE}
    expected_output = "key: enum_value\n"
    assert yaml.dump(data, Dumper=ConfigDumper) == expected_output
