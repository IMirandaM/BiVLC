import json
import pandas as pd
import csv


test_root = "./source_data/SugarCrepe"
test_dict = {
        'add_obj'    : f'{test_root}/add_obj.json',
        'add_att'    : f'{test_root}/add_att.json',
        'replace_obj': f'{test_root}/replace_obj.json',
        'replace_att': f'{test_root}/replace_att.json',
        'replace_rel': f'{test_root}/replace_rel.json',
        'swap_obj'   : f'{test_root}/swap_obj.json',
        'swap_att'   : f'{test_root}/swap_att.json',
    }

test = pd.DataFrame(columns = ['filename', 'caption', 'negative_caption', 'type', 'subtype'])
for c, data_path in test_dict.items():
      df= json.load(open(data_path))
      df = pd.DataFrame.from_dict(df, orient='index')
      c_split = c.partition("_")
      df['type'] = c_split[0]
      df['subtype'] = c_split[2]
      test = pd.concat([test, df])


print(f'Number of instances {len(test)}')
test.to_csv('./source_data/SugarCrepe/test_SugarCrepe.csv', index = False, quotechar='"', quoting=csv.QUOTE_ALL)