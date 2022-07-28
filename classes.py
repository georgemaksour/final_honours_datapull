from config import *


class TRACKER:
    def __init__(self):
        self.market_dict = {}

    def add_new_market(self, list_market_id):
        if len(list_market_id) > 0:
            for market_id in list_market_id:
                self.market_dict[market_id] = []

    def delete_market(self, list_market_id):
        if len(list_market_id) > 0:
            for market_id in list_market_id:
                if market_id in self.market_dict:
                    list_of_df = self.market_dict[market_id]
                    if len(list_of_df) != 0:
                        market_df = pd.concat(list_of_df)
                        name_id = str(market_id).replace('.', '_')
                        market_df.to_csv(f'data_files/{name_id}.csv')
                    self.market_dict.pop(market_id)

    def update_market(self, market_id, frame: pd.DataFrame):
        self.market_dict[market_id].append(frame)

    def get_keys(self):
        return list(self.market_dict.keys())

    def debug(self):
        print(self.market_dict.keys())
