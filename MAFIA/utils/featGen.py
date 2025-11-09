#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         featGen.py
 Description:  Technical feature generation.
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy
import os
import sys
sys.path.append(".")
from .data_validator import get_index_data_file, get_stock_data_file

# Try to import talib, with fallback if not available
try:
    from talib import abstract
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib is not available. Technical indicators from TA-Lib will not work.")
    print("Please install TA-Lib: https://github.com/TA-Lib/ta-lib-python")
    # Create a dummy abstract module to avoid errors
    class DummyAbstract:
        class Function:
            def __init__(self, *args, **kwargs):
                raise ImportError("TA-Lib is not installed. Please install it to use technical indicators.")
    abstract = DummyAbstract()

class FeatureProcesser:
    """
    Preprocess the training data.
    """
    def __init__(self, config):
        self.config = config
    
    def preprocess_feat(self, data):
        data = self.gen_feat(data=data)
        data = self.scale_feat(data=data)
        data = self.process_finedata(data=data)

        """
        data: dict
        - train: pd.DataFrame
        - valid: pd.DataFrame
        - test: pd.DataFrame
        - bftrain: pd.DataFrame
        - extra_train: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
        - extra_valid: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
        - extra_test: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
        """
        return data
    
    def gen_feat(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        # ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
        self.rawColLst = list(data.columns)
        datax = copy.deepcopy(data)
        stock_lst = datax['stock'].unique()
        for indidx, sigIndicatorName in enumerate(list(self.config.tech_indicator_input_lst) + list(self.config.otherRef_indicator_lst)):
            if sigIndicatorName.split('-')[0] in ['DAILYRETURNS']: 
                continue
            ind_df = pd.DataFrame()
            for sigStockName in stock_lst:
                dataSig = copy.deepcopy(data[data['stock']==sigStockName])
                dataSig.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
                if sigIndicatorName == 'CHANGE':
                    temp = {}
                    # Generate the training features
                    cg_close_ay_last = np.array(dataSig['close'])[:-1]
                    data_len = len(dataSig)
                    for change_feat in self.config.use_features:
                        cg_ay = np.array(dataSig[change_feat])[1:]
                        cg_ay = np.divide(cg_ay, cg_close_ay_last, out=np.ones_like(cg_ay), where=cg_close_ay_last!=0)
                        cg_ay[cg_ay==0] = 1
                        cg_ay = cg_ay - 1 # -> mean=0
                        # fill the first day data
                        cg_ay = np.append([0], cg_ay, axis=0)
                        # Ensure all arrays have the same length as the original data
                        temp['{}_w{}'.format(change_feat, 1)] = cg_ay
                        for widx in range(2, self.config.window_size+1):
                            # Ensure windowed array has same length as original data
                            if widx-1 < len(cg_ay):
                                windowed = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
                            else:
                                # If window size exceeds data length, pad with zeros
                                windowed = np.zeros(data_len)
                            # Trim or pad to match data length
                            if len(windowed) > data_len:
                                windowed = windowed[:data_len]
                            elif len(windowed) < data_len:
                                windowed = np.append(windowed, np.zeros(data_len - len(windowed)), axis=0)
                            temp['{}_w{}'.format(change_feat, widx)] = windowed

                # {indicator name}-{window}-{output field}-{input field}
                elif (sigIndicatorName in self.config.tech_indicator_talib_lst) or (sigIndicatorName in self.config.otherRef_indicator_lst):
                    indNameLst = sigIndicatorName.split('-')
                    iname = indNameLst[0] if len(indNameLst) > 0 else sigIndicatorName
                    
                    # Handle MA indicator with fallback if TA-Lib not available
                    if iname == 'MA' and not TALIB_AVAILABLE:
                        # Fallback: Use pandas rolling mean
                        if len(indNameLst) == 2:
                            window_size = int(indNameLst[1])
                        else:
                            window_size = 5  # default
                        ma_series = pd.Series(dataSig['close']).rolling(window=window_size, min_periods=1).mean()
                        temp = {sigIndicatorName: ma_series.values}
                    elif not TALIB_AVAILABLE:
                        raise ImportError("TA-Lib is required for indicator {} but is not installed. Please install TA-Lib.".format(sigIndicatorName))
                    else:
                        indFunc = abstract.Function(iname)
                        output_fields = indFunc.output_names
                        if 'price' in indFunc.input_names.keys():
                            ori_ifield = indFunc.input_names['price']
                        if 'prices' in indFunc.input_names.keys():
                            ori_ifield = indFunc.input_names['prices']
                        
                        if len(indNameLst) == 1:
                            iname = sigIndicatorName
                            window_size = None
                            ifield = ori_ifield
                            ofield = None
                        elif len(indNameLst) == 2:
                            iname = indNameLst[0]
                            if indNameLst[1] == 'None':
                                window_size = None
                            else:
                                window_size = int(indNameLst[1])
                            ofield = None
                            ifield = ori_ifield
                        elif len(indNameLst) == 3:
                            iname = indNameLst[0]
                            if indNameLst[1] == 'None':
                                window_size = None
                            else:
                                window_size = int(indNameLst[1])
                            if indNameLst[2] == 'None':
                                ofield = None
                            else:
                                ofield = indNameLst[2]
                            ifield = ori_ifield
                        elif len(indNameLst) == 4:
                            iname = indNameLst[0]
                            if indNameLst[1] == 'None':
                                window_size = None
                            else:
                                window_size = int(indNameLst[1])
                            if indNameLst[2] == 'None':
                                ofield = None
                            else:
                                ofield = indNameLst[2]
                            if indNameLst[3] == 'None':
                                ifield = ori_ifield
                            else:
                                ifield = indNameLst[3]   
                        else:
                            raise ValueError("Unexpect indicator {}".format(sigIndicatorName))
                        
                        if iname in ['OBV']:
                            ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']])
                        elif 'price' in indFunc.input_names.keys():
                            ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']], timeperiod=window_size, price=ifield)
                        elif 'prices' in indFunc.input_names.keys():
                            ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']], timeperiod=window_size, prices=ifield)
                        else:
                            raise ValueError("Invalid input fields: {}".format(indFunc.input_names))

                        if len(output_fields) == 1:
                            temp = {sigIndicatorName: np.array(ind_val.values)}
                        else:
                            if ofield is None:
                                if sigIndicatorName == 'MACD':
                                    temp = {sigIndicatorName: np.array(ind_val['macd'])}
                                elif sigIndicatorName == 'AROON':
                                    temp = {'AROONDOWN': np.array(ind_val['aroondown']), 'AROONUP': np.array(ind_val['aroonup'])}
                                elif sigIndicatorName == 'BBANDS':
                                    temp = {'BOLLUP': np.array(ind_val['upperband']), 'BOLLMID': np.array(ind_val['middleband']), 'BOLLLOW': np.array(ind_val['lowerband'])}
                                else:
                                    temp = {sigIndicatorName: np.array(ind_val[sorted(list(ind_val.keys()))[0]])}
                            else:
                                temp = {sigIndicatorName: np.array(ind_val[ofield])}

                else:
                    raise ValueError("Please specify the category of the indicator: {}".format(sigIndicatorName))               
                
                temp = pd.DataFrame(temp)
                temp['stock'] = sigStockName
                temp['date'] = np.array(dataSig['date'])
                ind_df = pd.concat([ind_df, temp], axis=0, join='outer')
            datax = pd.merge(datax, ind_df, how='outer', on=['stock', 'date'])
        
        datax.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        cur_cols =list(datax.columns)
        self.techIndicatorLst = sorted(list(set(cur_cols) - set(self.rawColLst) - set(self.config.otherRef_indicator_lst)))
        return datax

    def scale_feat(self, data):
        data['date'] = pd.to_datetime(data['date'])
        datax = copy.deepcopy(data)

        # covariance calculation
        if self.config.enable_cov_features:
            datax.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
            datax.index = datax.date.factorize()[0]
            cov_lst = []
            date_lst = []
            for idx in range(self.config.cov_lookback, datax['date'].nunique()):
                sigPeriodData = datax.loc[idx-self.config.cov_lookback:idx, :]
                sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
                sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
                sigPeriodReturn = sigPeriodClose.pct_change().dropna()
                covs = sigPeriodReturn.cov().values 
                cov_lst.append(covs)
                date_lst.append(datax.loc[idx, 'date'].values[0])
            
            cov_pd = pd.DataFrame({'date': date_lst, 'cov': cov_lst})
            datax = pd.merge(datax, cov_pd, how='inner', on=['date'])

        # [t-T, t-T+1, .., t-1, t]
        if 'DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback) in self.config.otherRef_indicator_lst:
            r_lst = []
            stockNo_lst = []
            date_lst = []
            datax.sort_values(['date', 'stock'], ascending=True, inplace=True)
            datax.reset_index(drop=True, inplace=True)
            datax.index = datax.date.factorize()[0]    
            for idx in range(self.config.dailyRetun_lookback, datax['date'].nunique()):
                sigPeriodData = datax.loc[idx-self.config.dailyRetun_lookback:idx, :][['date', 'stock', 'close']]
                sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
                sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
                sigPeriodReturn = sigPeriodClose.pct_change(fill_method=None).dropna() # without percentage
                sigPeriodReturn.sort_values(['date'], ascending=True, inplace=True)
                sigStockName_lst = np.array(sigPeriodReturn.columns)
                stockNo_lst = stockNo_lst + list(sigStockName_lst)
                r_lst = r_lst + list(np.transpose(sigPeriodReturn.values))
                date_lst = date_lst + [datax.loc[idx, 'date'].values[0]] * len(sigStockName_lst)
            r_pd = pd.DataFrame({'date': date_lst, 'stock': stockNo_lst, 'DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback): r_lst})
            datax = pd.merge(datax, r_pd, how='inner', on=['date', 'stock'])

        datax.reset_index(drop=True, inplace=True)
        if self.config.test_date_end is None:
            if self.config.valid_date_end is None:
                data_date_end = self.config.train_date_end
            else:
                data_date_end = self.config.valid_date_end
        else:
            data_date_end = self.config.test_date_end

        data_bftrain = copy.deepcopy(datax[datax['date'] < self.config.train_date_start][['date', 'stock', 'DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)]])
        data_bftrain = data_bftrain.dropna(axis=0, how='any')

        datax = copy.deepcopy(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= data_date_end)])
        datax.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True) 
        
        for sigIndicatorName in self.techIndicatorLst:
            # Feature normalization
            nan_cnt = len(np.argwhere(np.isnan(np.array(datax[sigIndicatorName]))))
            inf_cnt = len(np.argwhere(np.isinf(np.array(datax[sigIndicatorName]))))
            if (nan_cnt > 0) or (inf_cnt > 0):
                raise ValueError("Indicator: {}, nan count: {}, inf count: {}".format(sigIndicatorName, nan_cnt, inf_cnt))
            if (sigIndicatorName in ['CHANGELOGCLOSE', 'cov']) or ('close_w' in sigIndicatorName) or ('open_w' in sigIndicatorName) or ('high_w' in sigIndicatorName) or ('low_w' in sigIndicatorName) or ('volume_w' in sigIndicatorName):
                # No need to be normalized.
                continue
            train_ay = np.array(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= self.config.train_date_end)][sigIndicatorName])
            ind_mean = np.mean(train_ay)
            ind_std = np.std(train_ay, ddof=1)
            datax[sigIndicatorName] = (np.array(datax[sigIndicatorName]) - ind_mean) / ind_std
        if self.config.enable_cov_features:
            self.techIndicatorLst = list(self.techIndicatorLst) + ['cov']
        cols_order = list(self.rawColLst) + list(self.config.otherRef_indicator_lst) + list(sorted(self.techIndicatorLst)) 
        datax = datax[cols_order]
        
        dataset_dict = {}
        train_data = copy.deepcopy(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= self.config.train_date_end)])
        train_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        dataset_dict['train'] = train_data
        
        if (self.config.valid_date_start is not None) and (self.config.valid_date_end is not None):
            valid_data = copy.deepcopy(datax[(datax['date'] >= self.config.valid_date_start) & (datax['date'] <= self.config.valid_date_end)])
            valid_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
            dataset_dict['valid'] = valid_data

        if (self.config.test_date_start is not None) and (self.config.test_date_end is not None):
            test_data = copy.deepcopy(datax[(datax['date'] >= self.config.test_date_start) & (datax['date'] <= self.config.test_date_end)])
            test_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
            dataset_dict['test'] = test_data

        data_bftrain.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        dataset_dict['bftrain'] = data_bftrain

        print(datax)
        # ['date', 'stock', 'open', 'high', 'low', 'close'] + [{technical_indicators}]
        return dataset_dict

    def process_finedata(self, data):
        # Preprocess the data for the market observer.
        # Fine market data
        fine_mkt_data = self.gen_market_feat(freq=self.config.finefreq)
        # Fine stock data
        fine_stock_data = self.gen_fine_stock_feat()

        # Train
        daily_date_lst = data['train']['date'].unique()
        extra_train_data = {}

        fmd_train = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= self.config.train_date_start) & (fine_mkt_data['date'] <= self.config.train_date_end)])
        fmd_train.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
        extra_train_data['fine_market'] = fmd_train
        fmd_date_lst = fmd_train['date'].unique()
        # Use intersection of dates to avoid mismatch
        common_dates = set(fmd_date_lst) & set(daily_date_lst)
        if len(common_dates) == 0:
            raise ValueError("[Train, fine market] | No common dates between fine market data and daily data")
        if len(set(fmd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fmd_date_lst)) > 0:
            print("Warning: [Train, fine market] | Date mismatch - using intersection. Fine market dates: {}, Daily dates: {}, Common: {}".format(len(fmd_date_lst), len(daily_date_lst), len(common_dates)))

            fsd_train = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= self.config.train_date_start) & (fine_stock_data['date'] <= self.config.train_date_end)])
            fsd_train.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
            extra_train_data['fine_stock'] = fsd_train
            fsd_date_lst = fsd_train['date'].unique()
            # Use intersection of dates to avoid mismatch
            common_dates = set(fsd_date_lst) & set(daily_date_lst)
            if len(common_dates) == 0:
                raise ValueError("[Train, fine stock] | No common dates between fine stock data and daily data")
            if len(set(fsd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fsd_date_lst)) > 0:
                print("Warning: [Train, fine stock] | Date mismatch - using intersection. Fine stock dates: {}, Daily dates: {}, Common: {}".format(len(fsd_date_lst), len(daily_date_lst), len(common_dates)))
        data['extra_train'] = extra_train_data

        # Valid
        if (self.config.valid_date_start is not None) and (self.config.valid_date_end is not None):
            daily_date_lst = data['valid']['date'].unique()
            extra_valid_data = {}

            fmd_valid = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= self.config.valid_date_start) & (fine_mkt_data['date'] <= self.config.valid_date_end)])
            fmd_valid.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
            extra_valid_data['fine_market'] = fmd_valid
            fmd_date_lst = fmd_valid['date'].unique()
            # Use intersection of dates to avoid mismatch
            common_dates = set(fmd_date_lst) & set(daily_date_lst)
            if len(common_dates) == 0:
                raise ValueError("[Valid, fine market] | No common dates between fine market data and daily data")
            if len(set(fmd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fmd_date_lst)) > 0:
                print("Warning: [Valid, fine market] | Date mismatch - using intersection. Fine market dates: {}, Daily dates: {}, Common: {}".format(len(fmd_date_lst), len(daily_date_lst), len(common_dates)))

            fsd_valid = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= self.config.valid_date_start) & (fine_stock_data['date'] <= self.config.valid_date_end)])
            fsd_valid.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
            extra_valid_data['fine_stock'] = fsd_valid
            fsd_date_lst = fsd_valid['date'].unique()
            # Use intersection of dates to avoid mismatch
            common_dates = set(fsd_date_lst) & set(daily_date_lst)
            if len(common_dates) == 0:
                raise ValueError("[Valid, fine stock] | No common dates between fine stock data and daily data")
            if len(set(fsd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fsd_date_lst)) > 0:
                print("Warning: [Valid, fine stock] | Date mismatch - using intersection. Fine stock dates: {}, Daily dates: {}, Common: {}".format(len(fsd_date_lst), len(daily_date_lst), len(common_dates)))
            data['extra_valid'] = extra_valid_data

        # Test
        if (self.config.test_date_start is not None) and (self.config.test_date_end is not None):
            daily_date_lst = data['test']['date'].unique()
            extra_test_data = {}
            fmd_test = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= self.config.test_date_start) & (fine_mkt_data['date'] <= self.config.test_date_end)])
            fmd_test.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
            fmd_date_lst = fmd_test['date'].unique()
            
            # If no fine market data in test period, create from daily stock data
            if len(fmd_date_lst) == 0:
                print("Warning: [Test, fine market] | No fine market data in test period, creating from daily stock data")
                # Create fine market data from daily stock data by aggregating
                test_daily = data['test'].copy()
                # Group by date and calculate market-level features (average across stocks)
                market_features = []
                for date in daily_date_lst:
                    date_data = test_daily[test_daily['date'] == date]
                    if len(date_data) > 0:
                        # Calculate market-level features (average of all stocks)
                        market_row = {
                            'date': date,
                            'mkt_{}_close'.format(self.config.finefreq): date_data['close'].mean(),
                            'mkt_{}_ma'.format(self.config.finefreq): date_data['close'].mean(),  # Use close as MA fallback
                        }
                        # Add window features (simplified - use same value for all windows)
                        for feat in self.config.use_features:
                            for w in range(1, self.config.fine_window_size + 1):
                                market_row['mkt_{}_{}_w{}'.format(self.config.finefreq, feat, w)] = date_data[feat].mean() if feat in date_data.columns else 0.0
                        market_features.append(market_row)
                fmd_test = pd.DataFrame(market_features)
                fmd_test['date'] = pd.to_datetime(fmd_test['date'])
                fmd_date_lst = fmd_test['date'].unique()
            
            extra_test_data['fine_market'] = fmd_test
            # Use intersection of dates to avoid mismatch
            common_dates = set(fmd_date_lst) & set(daily_date_lst)
            if len(common_dates) == 0:
                raise ValueError("[Test, fine market] | No common dates between fine market data and daily data after fallback")
            if len(set(fmd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fmd_date_lst)) > 0:
                print("Warning: [Test, fine market] | Date mismatch - using intersection. Fine market dates: {}, Daily dates: {}, Common: {}".format(len(fmd_date_lst), len(daily_date_lst), len(common_dates)))

            fsd_test = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= self.config.test_date_start) & (fine_stock_data['date'] <= self.config.test_date_end)])
            fsd_test.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
            fsd_date_lst = fsd_test['date'].unique()
            
            # If no fine stock data in test period, create from daily stock data
            if len(fsd_date_lst) == 0:
                print("Warning: [Test, fine stock] | No fine stock data in test period, creating from daily stock data")
                # Create fine stock data from daily stock data
                test_daily = data['test'].copy()
                fine_stock_list = []
                for stock in test_daily['stock'].unique():
                    stock_data = test_daily[test_daily['stock'] == stock].sort_values('date')
                    for _, row in stock_data.iterrows():
                        fine_row = {
                            'stock': stock,
                            'date': row['date'],
                            'stock_{}_close'.format(self.config.finefreq): row['close'],
                            'stock_{}_ma'.format(self.config.finefreq): row['close'],  # Use close as MA fallback
                        }
                        # Add window features (simplified - use same value for all windows)
                        for feat in self.config.use_features:
                            for w in range(1, self.config.fine_window_size + 1):
                                fine_row['stock_{}_{}_w{}'.format(self.config.finefreq, feat, w)] = row[feat] if feat in row.index else 0.0
                        fine_stock_list.append(fine_row)
                fsd_test = pd.DataFrame(fine_stock_list)
                fsd_test['date'] = pd.to_datetime(fsd_test['date'])
                fsd_date_lst = fsd_test['date'].unique()
            
            extra_test_data['fine_stock'] = fsd_test
            # Use intersection of dates to avoid mismatch
            common_dates = set(fsd_date_lst) & set(daily_date_lst)
            if len(common_dates) == 0:
                raise ValueError("[Test, fine stock] | No common dates between fine stock data and daily data after fallback")
            if len(set(fsd_date_lst) - set(daily_date_lst)) > 0 or len(set(daily_date_lst) - set(fsd_date_lst)) > 0:
                print("Warning: [Test, fine stock] | Date mismatch - using intersection. Fine stock dates: {}, Daily dates: {}, Common: {}".format(len(fsd_date_lst), len(daily_date_lst), len(common_dates)))
            data['extra_test'] = extra_test_data
        return data

    def gen_market_feat(self, freq, daily_date_lst=None):
        # Get index data file with validation
        fpath, error_msg = get_index_data_file(self.config, freq=freq)
        isHasFineData = (freq != '1d' and fpath is not None and 
                        '{}_{}_index.csv'.format(self.config.market_name, freq) in fpath)
        
        if fpath is None:
            # Try fallback to 1d
            fpath, error_msg = get_index_data_file(self.config, freq='1d')
            isHasFineData = False
            if fpath is None:
                raise ValueError(f"Cannot load index data file. {error_msg}")
            print("Cannot find the {}-freq market data, will use 1d data instead.".format(freq))
        raw_data = pd.DataFrame(pd.read_csv(fpath, header=0, usecols=['date']+list(self.config.use_features)))
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        raw_data = raw_data.groupby(['date']).mean().reset_index(drop=False, inplace=False)
        raw_data.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)

        if freq == '1d':
            cur_winsize = self.config.window_size
        elif freq == '60m':
            cur_winsize = self.config.fine_window_size
        else:
            raise ValueError("Invalid freq[p1]: {}".format(freq))
        
        close_last_ay = np.array(raw_data['close'])[:-1]
        if not TALIB_AVAILABLE:
            # Simple moving average fallback using pandas
            ma_ay = pd.Series(raw_data['close']).rolling(window=cur_winsize+1, min_periods=1).mean().values
        else:
            ma_func = abstract.Function('ma')
            ma_ay = ma_func(np.array(raw_data['close']), timeperiod=cur_winsize+1)
        temp = {'date': np.array(raw_data['date']), 'mkt_{}_close'.format(freq): np.array(raw_data['close']), 'mkt_{}_ma'.format(freq): ma_ay}
        for change_feat in self.config.use_features:
            cg_ay = np.array(raw_data[change_feat])[1:]
            cg_ay = np.divide(cg_ay, close_last_ay, out=np.ones_like(cg_ay), where=close_last_ay!=0)
            cg_ay[cg_ay==0] = 1
            cg_ay = cg_ay - 1
            cg_ay = np.append([0], cg_ay, axis=0)
            cg_ay = cg_ay * self.config.feat_scaler
            temp['mkt_{}_{}_w{}'.format(freq, change_feat, 1)] = cg_ay
            for widx in range(2, cur_winsize+1):
                temp['mkt_{}_{}_w{}'.format(freq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
        mkt_pd = pd.DataFrame(temp)

        if freq == '60m':
            if isHasFineData:
                mkt_pd['time'] = mkt_pd['date'].apply(lambda x: x.strftime('%H:%M:%S')) # Get hour-min-sec
                mkt_pd = mkt_pd[mkt_pd['time']==self.config.market_close_time[self.config.market_name]][['date'] + self.config.finemkt_feat_cols_lst + ['mkt_{}_ma'.format(freq), 'mkt_{}_close'.format(freq)]] # Extract the datapoint of the market close time.
                mkt_pd['date'] = mkt_pd['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # Convert Year-Month-Day hh:mm:ss to Year-Month-Day
                mkt_pd['date'] = pd.to_datetime(mkt_pd['date'])
            else:
                mkt_pd['date'] = pd.to_datetime(mkt_pd['date'])
                mkt_pd = mkt_pd[['date'] + self.config.finemkt_feat_cols_lst + ['mkt_{}_ma'.format(freq), 'mkt_{}_close'.format(freq)]]
            mkt_pd.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
        elif freq == '1d':
            pass
        else:
            raise ValueError("Invalid freq[p2]: {}".format(freq))
        # columns: date, close/open/high/low_w{1-31/4}
        # The 60m fine market date only includes one datapoint per day (the datapoint is at the market close time), The other timepoints within a day are not included. 
        return mkt_pd

    def gen_fine_stock_feat(self, daily_date_lst=None):
        # For fine stock data, try to use the same stock data file or look for fine frequency version
        # First try to get fine frequency file
        if self.config.stock_data_file:
            # If specific file is set, use it
            fpath = os.path.join(self.config.dataDir, self.config.stock_data_file)
            isHasFineData = False  # Fine data would need separate file
        else:
            # Try fine frequency pattern first
            fpath = os.path.join(self.config.dataDir, '{}_{}_{}.csv'.format(self.config.market_name, self.config.topK, self.config.finefreq))
            isHasFineData = True
        
        if not os.path.exists(fpath):
            # Fallback to main stock data file
            fpath, error_msg = get_stock_data_file(self.config)
            isHasFineData = False
            if fpath is None:
                raise ValueError(f"Cannot load fine stock data file. {error_msg}")
            print("Cannot find the {}-freq stock data, will use main stock data file instead.".format(self.config.finefreq))
        else:
            # Validate the fine frequency file
            from .data_validator import validate_stock_data_file
            is_valid, error_msg = validate_stock_data_file(fpath)
            if not is_valid:
                # Fallback to main stock data file
                fpath, error_msg = get_stock_data_file(self.config)
                isHasFineData = False
                if fpath is None:
                    raise ValueError(f"Cannot load fine stock data file. Validation failed: {error_msg}")
                print("Fine stock data file validation failed, will use main stock data file instead.")
        
        raw_data = pd.DataFrame(pd.read_csv(fpath, header=0, usecols=['date', 'stock']+list(self.config.use_features)))
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        # raw_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        raw_data = raw_data.groupby(['date', 'stock']).mean().reset_index(drop=False, inplace=False)
        stock_lst = raw_data['stock'].unique()
        fine_data = pd.DataFrame()
        if TALIB_AVAILABLE:
            ma_func = abstract.Function('ma')
        for stock_id in stock_lst:
            dataSig = copy.deepcopy(raw_data[raw_data['stock']==stock_id])
            dataSig.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)

            if not TALIB_AVAILABLE:
                # Simple moving average fallback
                ma_ay = pd.Series(dataSig['close']).rolling(window=self.config.fine_window_size+1, min_periods=1).mean().values
            else:
                ma_ay = ma_func(np.array(dataSig['close']), timeperiod=self.config.fine_window_size+1)

            temp = {'date': np.array(dataSig['date']), 'stock_{}_close'.format(self.config.finefreq): np.array(dataSig['close']), 'stock_{}_ma'.format(self.config.finefreq): ma_ay}
            output_cols = ['date'] + self.config.finestock_feat_cols_lst + ['stock_{}_ma'.format(self.config.finefreq), 'stock_{}_close'.format(self.config.finefreq)]
            if self.config.is_gen_dc_feat:
                dc_events = dc_feature_generation(data=np.array(dataSig['close']), dc_threshold=self.config.dc_threshold[0])
                temp['stock_{}_dc'.format(self.config.finefreq)] = dc_events 
                output_cols = output_cols + ['stock_{}_dc'.format(self.config.finefreq)]
            close_last_ay = np.array(dataSig['close'])[:-1]
            # date, stock, close_w1, close_w2, .., open_w1, ..., close/open/high/low_w{1-4}
            for change_feat in self.config.use_features:
                cg_ay = np.array(dataSig[change_feat])[1:]
                cg_ay = np.divide(cg_ay, close_last_ay, out=np.ones_like(cg_ay), where=close_last_ay!=0)
                cg_ay[cg_ay==0] = 1
                cg_ay = cg_ay - 1
                cg_ay = np.append([0], cg_ay, axis=0)
                cg_ay = cg_ay * self.config.feat_scaler
                temp['stock_{}_{}_w{}'.format(self.config.finefreq, change_feat, 1)] = cg_ay
                for widx in range(2, self.config.fine_window_size+1):
                    temp['stock_{}_{}_w{}'.format(self.config.finefreq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)

            temp = pd.DataFrame(temp)
            if isHasFineData:
                temp['time'] = temp['date'].apply(lambda x: x.strftime('%H:%M:%S')) # Get hour-min-sec
                temp = temp[temp['time']==self.config.market_close_time[self.config.market_name]][output_cols] # Extract the datapoint of the market close time.
                temp['date'] = temp['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # Convert Year-Month-Day hh:mm:ss to Year-Month-Day
            else:
                temp = temp[output_cols]
            temp.reset_index(drop=True, inplace=True)
            temp['stock'] = stock_id
            fine_data = pd.concat([fine_data, temp], axis=0, join='outer')
        fine_data['date'] = pd.to_datetime(fine_data['date'])
        fine_data.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    
        # columns: stock, date, close/open/high/low_w{1-4}
        # The date only includes one datapoint per day (the datapoint is at the market close time), The other timepoints within a day are not included. 
        return fine_data

def dc_feature_generation(data, dc_threshold):
    # Directional Change (DC) implementation.
    dc_event_lst = [True] 

    ph = data[0]
    pl = data[0]
    # Training dataset DC patterns
    for idx in range(1, len(data)):
        if dc_event_lst[-1]:
            if data[idx] <= (ph * (1 - dc_threshold)):
                dc_event_lst.append(False) # Downturn Event
                pl= data[idx]
            else:
                dc_event_lst.append(dc_event_lst[-1]) # No DC pattern
                if ph < data[idx]:
                    ph = data[idx]
        else:
            if data[idx] >= (pl * (1 + dc_threshold)):
                dc_event_lst.append(True)  # Upturn Event
                ph = data[idx]
            else:
                dc_event_lst.append(dc_event_lst[-1])  # No DC pattern
                if pl > data[idx]:
                    pl = data[idx]
    # Uptrend event: True
    # Downtrend evvent: False
    return dc_event_lst
