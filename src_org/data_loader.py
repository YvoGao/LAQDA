
from torch.utils.data import Sampler
from typing import List, Iterable
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
import json
from tqdm import tqdm


def get_data(path, mode='json'):
    result = []
    with open(path, 'r', encoding='utf-8') as src:
        if mode == 'json':
            for line in tqdm(src):
                line = json.loads(line)
                result.append(line)
        else:
            for line in tqdm(src):
                line = line.split('\n')[0]
                result.append(line)
    return result
        


class KShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(KShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_set, query_set = [], []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                episode_labels = []
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        tmp = {"text": s['text'], "label": s['class_name']}
                        support_set.append(tmp)
                    episode_labels.append(s['class_name'])
                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        tmp = {"text": q['text'], "label": q['class_name']}
                        query_set.append(tmp)
                    
            yield np.stack(support_set), np.stack(query_set), episode_labels


class MyDataset(Dataset):
    def __init__(self, path):
        """Dataset class representing FewAsp dataset
        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        
        self.df = pd.DataFrame(self.index_subset(path))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # 这个的数据集处理是将label按照str名字读取的
        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
        # Create dicts
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        label = self.datasetid_to_class_id[item]
        text = self.df['text'][item]

        return text, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(path):
        """Index a subset by looping through all of its files and recording relevant information.
        # Arguments
            subset: Name of the subset
        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        texts = []
        print('Indexing {}...'.format(path))
        
        datas = get_data(path)
        for line in tqdm(datas):      
            texts.append({
                    'text': line["sentence"],
                    'class_name': line["label"]
            })
        return texts



def write_data(datas, path):
    with open(path, 'w') as fout:
        for line in datas:
            fout.write("%s\n" % json.dumps(line, ensure_ascii=False))

def get_label_dict(args):

    _hwu64_label_dict = {'audio/volume_down': 0, 'audio/volume_mute': 1, 'audio/volume_up': 2, 'calendar/query': 3, 'calendar/remove': 4, 'calendar/set': 5, 'email/addcontact': 6, 'email/query': 7, 'email/querycontact': 8, 'email/sendemail': 9, 'recommendation/events': 10, 'recommendation/locations': 11, 'recommendation/movies': 12, 'takeaway/order': 13, 'takeaway/query': 14, 'transport/query': 15, 'transport/taxi': 16, 'transport/ticket': 17, 'transport/traffic': 18, 'alarm/query': 19, 'alarm/remove': 20, 'alarm/set': 21, 'general/affirm': 22, 'general/commandstop': 23, 'general/confirm': 24, 'general/dontcare': 25, 'general/explain': 26, 'general/joke': 27, 'general/negate': 28, 'general/praise': 29,
                         'general/quirky': 30, 'general/repeat': 31, 'iot/cleaning': 32, 'iot/coffee': 33, 'iot/hue_lightchange': 34, 'iot/hue_lightdim': 35, 'iot/hue_lightoff': 36, 'iot/hue_lighton': 37, 'iot/hue_lightup': 38, 'iot/wemo_off': 39, 'iot/wemo_on': 40, 'qa/currency': 41, 'qa/definition': 42, 'qa/factoid': 43, 'qa/maths': 44, 'qa/stock': 45, 'social/post': 46, 'social/query': 47, 'weather/query': 48, 'cooking/recipe': 49, 'datetime/convert': 50, 'datetime/query': 51, 'lists/createoradd': 52, 'lists/query': 53, 'lists/remove': 54, 'music/likeness': 55, 'music/query': 56, 'music/settings': 57, 'news/query': 58, 'play/audiobook': 59, 'play/game': 60, 'play/music': 61, 'play/podcasts': 62, 'play/radio': 63}

    _liu_label_dict = {'post': 0, 'locations': 1, 'movies': 2, 'volume_mute': 3, 'radio': 4, 'audiobook': 5, 'stock': 6, 'events': 7, 'recipe': 8, 'game': 9, 'hue_lightdim': 10, 'set': 11, 'traffic': 12, 'definition': 13, 'joke': 14, 'wemo_off': 15, 'commandstop': 16, 'cleaning': 17, 'factoid': 18, 'negate': 19, 'currency': 20, 'hue_lighton': 21, 'coffee': 22, 'confirm': 23, 'wemo_on': 24, 'maths': 25, 'hue_lightup': 26,
                       'likeness': 27, 'createoradd': 28, 'querycontact': 29, 'repeat': 30, 'hue_lightchange': 31, 'sendemail': 32, 'order': 33, 'ticket': 34, 'convert': 35, 'hue_lightoff': 36, 'podcasts': 37, 'volume_up': 38, 'taxi': 39, 'settings': 40, 'dontcare': 41, 'remove': 42, 'explain': 43, 'dislikeness': 44, 'addcontact': 45, 'volume_down': 46, 'affirm': 47, 'praise': 48, 'greet': 49, 'quirky': 50, 'music': 51, 'query': 52, 'volume_other': 53}

    _reuters_label_dict = {
        'acquisition': 0,
        'aluminium': 1,
        'trade deficit': 2,
        'cocoa': 3,
        'coffee': 4,
        'copper': 5,
        'cotton': 6,
        'inflation': 7,
        'oil': 8,
        'profit': 9,
        'gdp': 10,
        'gold': 11,
        'grain': 12,
        'rate': 13,
        'industrial production': 14,
        'steel': 15,
        'unemployment': 16,
        'cattle': 17,
        'treasury bank': 18,
        'money supply': 19,
        'gas': 20,
        'orange': 21,
        'reserves': 22,
        'retail': 23,
        'rubber': 24,
        'ship': 25,
        'sugar': 26,
        'tin': 27,
        'tariffs': 28,
        'oils and fats tax': 29,
        'producer price wholesale': 30
    }

    # _20news_label_dict = {
    #     'talk.politics.mideast': 0,
    #     'sci.space': 1,
    #     'misc.forsale': 2,
    #     'talk.politics.misc': 3,
    #     'comp.graphics': 4,
    #     'sci.crypt': 5,
    #     'comp.windows.x': 6,
    #     'comp.os.ms-windows.misc': 7,
    #     'talk.politics.guns': 8,
    #     'talk.religion.misc': 9,
    #     'rec.autos': 10,
    #     'sci.med': 11,
    #     'comp.sys.mac.hardware': 12,
    #     'sci.electronics': 13,
    #     'rec.sport.hockey': 14,
    #     'alt.atheism': 15,
    #     'rec.motorcycles': 16,
    #     'comp.sys.ibm.pc.hardware': 17,
    #     'rec.sport.baseball': 18,
    #     'soc.religion.christian': 19,
    # }

    _20news_label_dict = {
        'talk politics mideast': 0,
        'science space': 1,
        'misc forsale': 2,
        'talk politics misc': 3,
        'computer graphics': 4,
        'science  encryption  encrypt secret': 5,
        'computer windows x': 6,
        'computer os ms windows misc': 7,
        'talk politics guns': 8,
        'talk religion misc': 9,
        'rec autos': 10,
        'science med chemistry medical science medicine': 11,
        'computer sys mac hardware': 12,
        'science electronics': 13,
        'rec sport hockey': 14,
        'alt atheism': 15,
        'rec motorcycles': 16,
        'computer system ibm pc hardware': 17,
        'rec sport baseball': 18,
        'soc religion christian': 19,
    }

    # _amazon_label_dict = {
    #     'Amazon_Instant_Video': 0,
    #     'Apps_for_Android': 1,
    #     'Automotive': 2,
    #     'Baby': 3,
    #     'Beauty': 4,
    #     'Books': 5,
    #     'CDs_and_Vinyl': 6,
    #     'Cell_Phones_and_Accessories': 7,
    #     'Clothing_Shoes_and_Jewelry': 8,
    #     'Digital_Music': 9,
    #     'Electronics': 10,
    #     'Grocery_and_Gourmet_Food': 11,
    #     'Health_and_Personal_Care': 12,
    #     'Home_and_Kitchen': 13,
    #     'Kindle_Store': 14,
    #     'Movies_and_TV': 15,
    #     'Musical_Instruments': 16,
    #     'Office_Products': 17,
    #     'Patio_Lawn_and_Garden': 18,
    #     'Pet_Supplies': 19,
    #     'Sports_and_Outdoors': 20,
    #     'Tools_and_Home_Improvement': 21,
    #     'Toys_and_Games': 22,
    #     'Video_Games': 23
    # }

    # 多标签分类
    _amazon_label_dict = {
        'Amazon Instant Video': 0,
        'Apps for Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs and Vinyl': 6,
        'Cell Phones and Accessories': 7,
        'Clothing Shoes and Jewelry': 8,
        'Digital Music': 9,
        'Electronics': 10,
        'Grocery and Gourmet Food': 11,
        'Health and Personal Care': 12,
        'Home and Kitchen': 13,
        'Kindle Store': 14,
        'Movies and TV': 15,
        'Musical Instruments': 16,
        'Office Products': 17,
        'Patio Lawn and Garden': 18,
        'Pet Supplies': 19,
        'Sports and Outdoors': 20,
        'Tools and Home Improvement': 21,
        'Toys and Games': 22,
        'Video Games': 23
    }

    # _huffpost_label_dict = {'POLITICS': 0, 'WELLNESS': 1, 'ENTERTAINMENT': 2, 'TRAVEL': 3, 'STYLE & BEAUTY': 4,
    #                         'PARENTING': 5, 'HEALTHY LIVING': 6, 'QUEER VOICES': 7, 'FOOD & DRINK': 8, 'BUSINESS': 9,
    #                         'COMEDY': 10, 'SPORTS': 11, 'BLACK VOICES': 12, 'HOME & LIVING': 13, 'PARENTS': 14,
    #                         'THE WORLDPOST': 15, 'WEDDINGS': 16, 'WOMEN': 17, 'IMPACT': 18, 'DIVORCE': 19, 'CRIME': 20,
    #                         'MEDIA': 21, 'WEIRD NEWS': 22, 'GREEN': 23, 'WORLDPOST': 24, 'RELIGION': 25, 'STYLE': 26,
    #                         'SCIENCE': 27, 'WORLD NEWS': 28, 'TASTE': 29, 'TECH': 30, 'MONEY': 31, 'ARTS': 32,
    #                         'FIFTY': 33,
    #                         'GOOD NEWS': 34, 'ARTS & CULTURE': 35, 'ENVIRONMENT': 36, 'COLLEGE': 37,
    #                         'LATINO VOICES': 38,
    #                         'CULTURE & ARTS': 39, 'EDUCATION': 40}

    _huffpost_label_dict = {'politics': 0, 'wellness': 1, 'entertainment': 2, 'travel': 3, 'style and beauty': 4,
                            'parenting': 5, 'healthy living': 6, 'queer voices': 7, 'food': 8, 'business': 9,
                            'comedy': 10, 'sports': 11, 'black voices': 12, 'home and living': 13, 'parents': 14,
                            'the worldpost': 15, 'weddings': 16, 'women': 17, 'impact': 18, 'divorce': 19, 'crime': 20,
                            'media': 21, 'weird': 22, 'green': 23, 'world post': 24, 'religion': 25, 'style': 26,
                            'science': 27, 'world': 28, 'taste': 29, 'technology': 30, 'money': 31, 'arts': 32,
                            'fifty': 33,
                            'good': 34, 'arts and culture': 35, 'environment': 36, 'college': 37,
                            'latino voices': 38,
                            'culture and arts': 39, 'education': 40}

    # _banking77_label_dict = {'card_arrival': 0, 'card_linking': 1, 'exchange_rate': 2,
    #                          'card_payment_wrong_exchange_rate': 3, 'extra_charge_on_statement': 4,
    #                          'pending_cash_withdrawal': 5, 'fiat_currency_support': 6, 'card_delivery_estimate': 7,
    #                          'automatic_top_up': 8, 'card_not_working': 9, 'exchange_via_app': 10,
    #                          'lost_or_stolen_card': 11, 'age_limit': 12, 'pin_blocked': 13,
    #                          'contactless_not_working': 14,
    #                          'top_up_by_bank_transfer_charge': 15, 'pending_top_up': 16, 'cancel_transfer': 17,
    #                          'top_up_limits': 18, 'wrong_amount_of_cash_received': 19, 'card_payment_fee_charged': 20,
    #                          'transfer_not_received_by_recipient': 21, 'supported_cards_and_currencies': 22,
    #                          'getting_virtual_card': 23, 'card_acceptance': 24, 'top_up_reverted': 25,
    #                          'balance_not_updated_after_cheque_or_cash_deposit': 26, 'card_payment_not_recognised': 27,
    #                          'edit_personal_details': 28, 'why_verify_identity': 29, 'unable_to_verify_identity': 30,
    #                          'get_physical_card': 31, 'visa_or_mastercard': 32, 'topping_up_by_card': 33,
    #                          'disposable_card_limits': 34, 'compromised_card': 35, 'atm_support': 36,
    #                          'direct_debit_payment_not_recognised': 37, 'passcode_forgotten': 38,
    #                          'declined_cash_withdrawal': 39, 'pending_card_payment': 40, 'lost_or_stolen_phone': 41,
    #                          'request_refund': 42, 'declined_transfer': 43, 'Refund_not_showing_up': 44,
    #                          'declined_card_payment': 45, 'pending_transfer': 46, 'terminate_account': 47,
    #                          'card_swallowed': 48, 'transaction_charged_twice': 49, 'verify_source_of_funds': 50,
    #                          'transfer_timing': 51, 'reverted_card_payment?': 52, 'change_pin': 53,
    #                          'beneficiary_not_allowed': 54, 'transfer_fee_charged': 55, 'receiving_money': 56,
    #                          'failed_transfer': 57, 'transfer_into_account': 58, 'verify_top_up': 59,
    #                          'getting_spare_card': 60, 'top_up_by_cash_or_cheque': 61, 'order_physical_card': 62,
    #                          'virtual_card_not_working': 63, 'wrong_exchange_rate_for_cash_withdrawal': 64,
    #                          'get_disposable_virtual_card': 65, 'top_up_failed': 66,
    #                          'balance_not_updated_after_bank_transfer': 67, 'cash_withdrawal_not_recognised': 68,
    #                          'exchange_charge': 69, 'top_up_by_card_charge': 70, 'activate_my_card': 71,
    #                          'cash_withdrawal_charge': 72, 'card_about_to_expire': 73, 'apple_pay_or_google_pay': 74,
    #                          'verify_my_identity': 75, 'country_support': 76}

    _banking77_label_dict = {'card arrival': 0, 'card linking': 1, 'exchange rate': 2,
                             'card payment wrong exchange rate': 3, 'extra charge on statement': 4,
                             'pending cash withdrawal': 5, 'fiat currency support': 6, 'card delivery estimate': 7,
                             'automatic top up': 8, 'card not working': 9, 'exchange via app': 10,
                             'lost or stolen card': 11, 'age limit': 12, 'pin blocked': 13,
                             'contactless not working': 14,
                             'top up by bank transfer charge': 15, 'pending top up': 16, 'cancel transfer': 17,
                             'top up limits': 18, 'wrong amount of cash received': 19, 'card payment fee charged': 20,
                             'transfer not received by recipient': 21, 'supported cards and currencies': 22,
                             'getting virtual card': 23, 'card acceptance': 24, 'top up reverted': 25,
                             'balance not updated after cheque or cash deposit': 26, 'card payment not recognised': 27,
                             'edit personal details': 28, 'why verify identity': 29, 'unable to verify identity': 30,
                             'get physical card': 31, 'visa or mastercard': 32, 'topping up by card': 33,
                             'disposable card limits': 34, 'compromised card': 35, 'atm support': 36,
                             'direct debit payment not recognised': 37, 'passcode forgotten': 38,
                             'declined cash withdrawal': 39, 'pending card payment': 40, 'lost or stolen phone': 41,
                             'request refund': 42, 'declined transfer': 43, 'Refund not showing up': 44,
                             'declined card payment': 45, 'pending transfer': 46, 'terminate account': 47,
                             'card swallowed': 48, 'transaction charged twice': 49, 'verify source of funds': 50,
                             'transfer timing': 51, 'reverted card payment?': 52, 'change pin': 53,
                             'beneficiary not allowed': 54, 'transfer fee charged': 55, 'receiving money': 56,
                             'failed transfer': 57, 'transfer into account': 58, 'verify top up': 59,
                             'getting spare card': 60, 'top up by cash or cheque': 61, 'order physical card': 62,
                             'virtual card not working': 63, 'wrong exchange rate for cash withdrawal': 64,
                             'get disposable virtual card': 65, 'top up failed': 66,
                             'balance not updated after bank transfer': 67, 'cash withdrawal not recognised': 68,
                             'exchange charge': 69, 'top up by card charge': 70, 'activate my card': 71,
                             'cash withdrawal charge': 72, 'card about to expire': 73, 'apple pay or google pay': 74,
                             'verify my identity': 75, 'country support': 76}

    # _clinc150_label_dict = {'transfer': 0, 'transactions': 1, 'balance': 2, 'freeze_account': 3, 'pay_bill': 4,
    #                         'bill_balance': 5, 'bill_due': 6, 'interest_rate': 7, 'routing': 8, 'min_payment': 9,
    #                         'order_checks': 10, 'pin_change': 11, 'report_fraud': 12, 'account_blocked': 13,
    #                         'spending_history': 14, 'credit_score': 15, 'report_lost_card': 16, 'credit_limit': 17,
    #                         'rewards_balance': 18, 'new_card': 19, 'application_status': 20, 'card_declined': 21,
    #                         'international_fees': 22, 'apr': 23, 'redeem_rewards': 24, 'credit_limit_change': 25,
    #                         'damaged_card': 26, 'replacement_card_duration': 27, 'improve_credit_score': 28,
    #                         'expiration_date': 29, 'recipe': 30, 'restaurant_reviews': 31, 'calories': 32,
    #                         'nutrition_info': 33, 'restaurant_suggestion': 34, 'ingredients_list': 35,
    #                         'ingredient_substitution': 36, 'cook_time': 37, 'food_last': 38, 'meal_suggestion': 39,
    #                         'restaurant_reservation': 40, 'confirm_reservation': 41, 'how_busy': 42,
    #                         'cancel_reservation': 43, 'accept_reservations': 44, 'shopping_list': 45,
    #                         'shopping_list_update': 46, 'next_song': 47, 'play_music': 48, 'update_playlist': 49,
    #                         'todo_list': 50, 'todo_list_update': 51, 'calendar': 52, 'calendar_update': 53,
    #                         'what_song': 54,
    #                         'order': 55, 'order_status': 56, 'reminder': 57, 'reminder_update': 58, 'smart_home': 59,
    #                         'traffic': 60, 'directions': 61, 'gas': 62, 'gas_type': 63, 'distance': 64,
    #                         'current_location': 65, 'mpg': 66, 'oil_change_when': 67, 'oil_change_how': 68,
    #                         'jump_start': 69, 'uber': 70, 'schedule_maintenance': 71, 'last_maintenance': 72,
    #                         'tire_pressure': 73, 'tire_change': 74, 'book_flight': 75, 'book_hotel': 76,
    #                         'car_rental': 77,
    #                         'travel_suggestion': 78, 'travel_alert': 79, 'travel_notification': 80, 'carry_on': 81,
    #                         'timezone': 82, 'vaccines': 83, 'translate': 84, 'flight_status': 85,
    #                         'international_visa': 86,
    #                         'lost_luggage': 87, 'plug_type': 88, 'exchange_rate': 89, 'time': 90, 'alarm': 91,
    #                         'share_location': 92, 'find_phone': 93, 'weather': 94, 'text': 95, 'spelling': 96,
    #                         'make_call': 97, 'timer': 98, 'date': 99, 'calculator': 100, 'measurement_conversion': 101,
    #                         'flip_coin': 102, 'roll_dice': 103, 'definition': 104, 'direct_deposit': 105,
    #                         'pto_request': 106, 'taxes': 107, 'payday': 108, 'w2': 109, 'pto_balance': 110,
    #                         'pto_request_status': 111, 'next_holiday': 112, 'insurance': 113, 'insurance_change': 114,
    #                         'schedule_meeting': 115, 'pto_used': 116, 'meeting_schedule': 117, 'rollover_401k': 118,
    #                         'income': 119, 'greeting': 120, 'goodbye': 121, 'tell_joke': 122, 'where_are_you_from': 123,
    #                         'how_old_are_you': 124, 'what_is_your_name': 125, 'who_made_you': 126, 'thank_you': 127,
    #                         'what_can_i_ask_you': 128, 'what_are_your_hobbies': 129, 'do_you_have_pets': 130,
    #                         'are_you_a_bot': 131, 'meaning_of_life': 132, 'who_do_you_work_for': 133, 'fun_fact': 134,
    #                         'change_ai_name': 135, 'change_user_name': 136, 'cancel': 137, 'user_name': 138,
    #                         'reset_settings': 139, 'whisper_mode': 140, 'repeat': 141, 'no': 142, 'yes': 143,
    #                         'maybe': 144,
    #                         'change_language': 145, 'change_accent': 146, 'change_volume': 147, 'change_speed': 148,
    #                         'sync_device': 149}
    _clinc150_label_dict = {'transfer': 0, 'transactions': 1, 'balance': 2, 'freeze account': 3, 'pay bill': 4,
                            'bill balance': 5, 'bill due': 6, 'interest rate': 7, 'routing': 8, 'min payment': 9,
                            'order checks': 10, 'pin change': 11, 'report fraud': 12, 'account blocked': 13,
                            'spending history': 14, 'credit score': 15, 'report lost card': 16, 'credit limit': 17,
                            'rewards balance': 18, 'new card': 19, 'application status': 20, 'card declined': 21,
                            'international fees': 22, 'apr': 23, 'redeem rewards': 24, 'credit limit change': 25,
                            'damaged card': 26, 'replacement card duration': 27, 'improve credit score': 28,
                            'expiration date': 29, 'recipe': 30, 'restaurant reviews': 31, 'calories': 32,
                            'nutrition info': 33, 'restaurant suggestion': 34, 'ingredients list': 35,
                            'ingredient substitution': 36, 'cook time': 37, 'food last': 38, 'meal suggestion': 39,
                            'restaurant reservation': 40, 'confirm reservation': 41, 'how busy': 42,
                            'cancel reservation': 43, 'accept reservations': 44, 'shopping list': 45,
                            'shopping list update': 46, 'next song': 47, 'play music': 48, 'update playlist': 49,
                            'todo list': 50, 'todo list update': 51, 'calendar': 52, 'calendar update': 53,
                            'what song': 54,
                            'order': 55, 'order status': 56, 'reminder': 57, 'reminder update': 58, 'smart home': 59,
                            'traffic': 60, 'directions': 61, 'gas': 62, 'gas type': 63, 'distance': 64,
                            'current location': 65, 'mpg': 66, 'oil change when': 67, 'oil change how': 68,
                            'jump start': 69, 'uber': 70, 'schedule maintenance': 71, 'last maintenance': 72,
                            'tire pressure': 73, 'tire change': 74, 'book flight': 75, 'book hotel': 76,
                            'car rental': 77,
                            'travel suggestion': 78, 'travel alert': 79, 'travel notification': 80, 'carry on': 81,
                            'timezone': 82, 'vaccines': 83, 'translate': 84, 'flight status': 85,
                            'international visa': 86,
                            'lost luggage': 87, 'plug type': 88, 'exchange rate': 89, 'time': 90, 'alarm': 91,
                            'share location': 92, 'find phone': 93, 'weather': 94, 'text': 95, 'spelling': 96,
                            'make call': 97, 'timer': 98, 'date': 99, 'calculator': 100, 'measurement conversion': 101,
                            'flip coin': 102, 'roll dice': 103, 'definition': 104, 'direct deposit': 105,
                            'pto request': 106, 'taxes': 107, 'payday': 108, 'w2': 109, 'pto balance': 110,
                            'pto request status': 111, 'next holiday': 112, 'insurance': 113, 'insurance change': 114,
                            'schedule meeting': 115, 'pto used': 116, 'meeting schedule': 117, 'rollover 401k': 118,
                            'income': 119, 'greeting': 120, 'goodbye': 121, 'tell joke': 122, 'where are you from': 123,
                            'how old are you': 124, 'what is your name': 125, 'who made you': 126, 'thank you': 127,
                            'what can i ask you': 128, 'what are your hobbies': 129, 'do you have pets': 130,
                            'are you a bot': 131, 'meaning of life': 132, 'who do you work for': 133, 'fun fact': 134,
                            'change ai name': 135, 'change user name': 136, 'cancel': 137, 'user name': 138,
                            'reset settings': 139, 'whisper mode': 140, 'repeat': 141, 'no': 142, 'yes': 143,
                            'maybe': 144,
                            'change language': 145, 'change accent': 146, 'change volume': 147, 'change speed': 148,
                            'sync device': 149}
    if args.dataset == '20News' or args.dataset == '20newsgroup2':
        return _20news_label_dict
    elif args.dataset == 'Amazon' or args.dataset == 'amazon2':
        return _amazon_label_dict
    elif args.dataset == 'HuffPost':
        return _huffpost_label_dict
    elif args.dataset == 'Banking77':
        return _banking77_label_dict
    elif args.dataset == 'Clinc150':
        return _clinc150_label_dict
    elif args.dataset == 'Reuters':
        return _reuters_label_dict
    elif args.dataset == 'Liu':
        return _liu_label_dict
    elif args.dataset == 'Hwu64':
        return _hwu64_label_dict

