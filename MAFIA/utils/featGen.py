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
from .data_validator import get_index_data_file, get_stock_data_file, validate_stock_data_file

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
                    data_len = len(dataSig)
                    for change_feat in self.config.use_features:
                        feat_vals = np.array(dataSig[change_feat])
                        if len(feat_vals) <= 1:
                            cg_ay = np.zeros(data_len)
                        else:
                            prev_vals = feat_vals[:-1]
                            cur_vals = feat_vals[1:]
                            cg_ay = np.divide(cur_vals, prev_vals, out=np.ones_like(cur_vals, dtype=float), where=prev_vals!=0)
                            cg_ay[cg_ay==0] = 1
                            cg_ay = cg_ay - 1 # -> mean=0
                            cg_ay = np.append([0], cg_ay, axis=0)
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
        # Train-only normalization for market index indicators
        if hasattr(self.config, 'finemkt_indicator_cols') and len(self.config.finemkt_indicator_cols) > 0:
            norm_cols = [c for c in self.config.finemkt_indicator_cols if c in fmd_train.columns]
            train_stats = {}
            for c in norm_cols:
                vals = fmd_train[c].values
                mu = np.mean(vals)
                sigma = np.std(vals, ddof=1)
                sigma = sigma if sigma != 0 else 1.0
                train_stats[c] = (mu, sigma)
                fmd_train[c] = (fmd_train[c] - mu) / sigma
        extra_train_data['fine_market'] = fmd_train
        extra_train_data['market_index_feature_names'] = self.config.market_index_feature_names
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
            # Apply train-only normalization
            if hasattr(self, 'train_stats') and False:
                pass
            elif 'train_stats' in locals():
                for c, (mu, sigma) in train_stats.items():
                    if c in fmd_valid.columns:
                        fmd_valid[c] = (fmd_valid[c] - mu) / sigma
            extra_valid_data['fine_market'] = fmd_valid
            extra_valid_data['market_index_feature_names'] = self.config.market_index_feature_names
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
            extra_test_data['market_index_feature_names'] = self.config.market_index_feature_names
            # Apply train-only normalization
            if 'train_stats' in locals():
                for c, (mu, sigma) in train_stats.items():
                    if c in extra_test_data['fine_market'].columns:
                        extra_test_data['fine_market'][c] = (extra_test_data['fine_market'][c] - mu) / sigma
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
        # Try to get index data file first (for backward compatibility)
        fpath, error_msg = get_index_data_file(self.config, freq=freq)
        isHasFineData = (freq != '1d' and fpath is not None and 
                        '{}_{}_index.csv'.format(self.config.market_name, freq) in fpath)
        
        if fpath is None:
            # Try fallback to 1d index file
            fpath, error_msg = get_index_data_file(self.config, freq='1d')
            isHasFineData = False
            if fpath is None:
                # If no index file found, generate market features from stock data
                print("No index data file found. Generating market features from stock data instead.")
                return self._gen_market_feat_from_stock_data(freq)
            print("Cannot find the {}-freq market data, will use 1d data instead.".format(freq))
        
        # Load from index file
        raw_data = pd.DataFrame(pd.read_csv(fpath, header=0, usecols=['date']+list(self.config.use_features)))
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        # Normalize timezone: remove timezone info to match config dates (naive datetime)
        if raw_data['date'].dt.tz is not None:
            raw_data['date'] = raw_data['date'].dt.tz_localize(None)
        raw_data = raw_data.groupby(['date']).mean().reset_index(drop=False, inplace=False)
        raw_data.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)

        # For market index features, use mafia_T_w (e.g., 30) for windowed ΔOHLCV
        if hasattr(self.config, 'mafia_T_w') and isinstance(self.config.mafia_T_w, int):
            cur_winsize = self.config.mafia_T_w
        else:
            raise ValueError("Invalid freq[p1]: {}".format(freq))
        
        if not TALIB_AVAILABLE:
            # Simple moving average fallback using pandas
            ma_ay = pd.Series(raw_data['close']).rolling(window=cur_winsize+1, min_periods=1).mean().values
        else:
            ma_func = abstract.Function('ma')
            ma_ay = ma_func(np.array(raw_data['close']), timeperiod=cur_winsize+1)
        temp = {'date': np.array(raw_data['date']), 'mkt_{}_close'.format(freq): np.array(raw_data['close']), 'mkt_{}_ma'.format(freq): ma_ay}
        for change_feat in self.config.use_features:
            feat_vals = np.array(raw_data[change_feat])
            if len(feat_vals) <= 1:
                cg_ay = np.zeros(len(raw_data))
            else:
                prev_vals = feat_vals[:-1]
                cur_vals = feat_vals[1:]
                cg_ay = np.divide(cur_vals, prev_vals, out=np.ones_like(cur_vals, dtype=float), where=prev_vals!=0)
                cg_ay[cg_ay==0] = 1
                cg_ay = cg_ay - 1
                cg_ay = np.append([0], cg_ay, axis=0)
            cg_ay = cg_ay * self.config.feat_scaler
            temp['mkt_{}_{}_w{}'.format(freq, change_feat, 1)] = cg_ay
            for widx in range(2, cur_winsize+1):
                temp['mkt_{}_{}_w{}'.format(freq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
        mkt_pd = pd.DataFrame(temp)
        # Add market technical indicators (SMA20, RSI14, ATR14) and extended indicators
        try:
            close_vals = np.array(raw_data['close'])
            high_vals = np.array(raw_data['high']) if 'high' in raw_data.columns else close_vals
            low_vals = np.array(raw_data['low']) if 'low' in raw_data.columns else close_vals
            vol_vals = np.array(raw_data['volume']) if 'volume' in raw_data.columns else np.ones_like(close_vals)
            # SMA20 (already computed with cur_winsize+1; also add fixed 20)
            sma20 = pd.Series(close_vals).rolling(window=20, min_periods=1).mean().values
            # RSI14
            delta = np.diff(close_vals)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(alpha=1.0/14, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1.0/14, adjust=False).mean().values
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain, dtype=float), where=avg_loss!=0)
            rsi14 = 100 - (100 / (1 + rs))
            rsi14 = np.concatenate([[50.0], rsi14])
            if len(rsi14) < len(close_vals):
                rsi14 = np.pad(rsi14, (0, len(close_vals) - len(rsi14)), mode='edge')
            # ATR14
            tr = np.zeros(len(close_vals))
            for i in range(1, len(close_vals)):
                tr[i] = max(
                    high_vals[i] - low_vals[i],
                    abs(high_vals[i] - close_vals[i-1]),
                    abs(low_vals[i] - close_vals[i-1])
                )
            atr14 = pd.Series(tr).ewm(alpha=1.0/14, adjust=False).mean().values
            mkt_pd['mkt_{}_sma20'.format(freq)] = sma20
            mkt_pd['mkt_{}_rsi14'.format(freq)] = rsi14
            mkt_pd['mkt_{}_atr14'.format(freq)] = atr14
            # MACD(12,26,9) histogram
            ema12 = pd.Series(close_vals).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(close_vals).ewm(span=26, adjust=False).mean().values
            macd_line = ema12 - ema26
            macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            macd_hist = macd_line - macd_signal
            mkt_pd['mkt_{}_macd_hist'.format(freq)] = macd_hist
            # Bollinger Band Width(20,2)
            bb_mid = pd.Series(close_vals).rolling(window=20, min_periods=1).mean()
            bb_std = pd.Series(close_vals).rolling(window=20, min_periods=1).std(ddof=0)
            bb_up = bb_mid + 2 * bb_std
            bb_low = bb_mid - 2 * bb_std
            bb_width = (bb_up - bb_low) / (bb_mid.replace(0, np.nan)).replace(np.nan, 1.0)
            mkt_pd['mkt_{}_bb_width'.format(freq)] = np.nan_to_num(bb_width.values, nan=0.0, posinf=0.0, neginf=0.0)
            # Stochastic %K/%D(14,3)
            rolling_high14 = pd.Series(high_vals).rolling(window=14, min_periods=1).max().values
            rolling_low14 = pd.Series(low_vals).rolling(window=14, min_periods=1).min().values
            stoch_k = np.divide(close_vals - rolling_low14, rolling_high14 - rolling_low14, out=np.zeros_like(close_vals), where=(rolling_high14 - rolling_low14)!=0) * 100.0
            stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=1).mean().values
            mkt_pd['mkt_{}_stoch_k'.format(freq)] = stoch_k
            mkt_pd['mkt_{}_stoch_d'.format(freq)] = stoch_d
            # ADX(14)
            plus_dm = np.zeros(len(close_vals))
            minus_dm = np.zeros(len(close_vals))
            for i in range(1, len(close_vals)):
                up_move = high_vals[i] - high_vals[i-1]
                down_move = low_vals[i-1] - low_vals[i]
                plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
                minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr_series = pd.Series(tr)
            atr14_series = tr_series.ewm(alpha=1.0/14, adjust=False).mean()
            plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1.0/14, adjust=False).mean() / atr14_series.replace(0, np.nan)).replace(np.nan, 0.0)
            minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1.0/14, adjust=False).mean() / atr14_series.replace(0, np.nan)).replace(np.nan, 0.0)
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)).replace(np.nan, 0.0)
            adx14 = dx.ewm(alpha=1.0/14, adjust=False).mean().values
            mkt_pd['mkt_{}_adx14'.format(freq)] = adx14
            # OBV
            price_diff = np.diff(close_vals, prepend=close_vals[0])
            vol_sign = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
            obv = np.cumsum(vol_sign * vol_vals)
            mkt_pd['mkt_{}_obv'.format(freq)] = obv
            # MFI(14)
            typical_price = (high_vals + low_vals + close_vals) / 3.0
            tp_diff = np.diff(typical_price, prepend=typical_price[0])
            raw_mf = typical_price * vol_vals
            pos_mf = np.where(tp_diff > 0, raw_mf, 0.0)
            neg_mf = np.where(tp_diff < 0, raw_mf, 0.0)
            pos_mf14 = pd.Series(pos_mf).rolling(window=14, min_periods=1).sum()
            neg_mf14 = pd.Series(neg_mf).rolling(window=14, min_periods=1).sum()
            mfr = np.divide(pos_mf14, neg_mf14.replace(0, np.nan)).replace(np.nan, 1.0)
            mfi14 = 100 - (100 / (1 + mfr))
            mkt_pd['mkt_{}_mfi14'.format(freq)] = mfi14.values
            # CCI(20)
            sma_tp20 = pd.Series(typical_price).rolling(window=20, min_periods=1).mean()
            md20 = pd.Series(typical_price).rolling(window=20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            cci20 = (typical_price - sma_tp20) / (0.015 * md20.replace(0, np.nan)).replace(np.nan, 1.0)
            mkt_pd['mkt_{}_cci20'.format(freq)] = np.nan_to_num(cci20.values, nan=0.0, posinf=0.0, neginf=0.0)
            # Volatility std of returns (20)
            returns = pd.Series(close_vals).pct_change(fill_method=None).fillna(0.0)
            vol_std20 = returns.rolling(window=20, min_periods=1).std(ddof=0).values
            mkt_pd['mkt_{}_vol_std20'.format(freq)] = vol_std20
            # Drawdown (relative to rolling 60-day peak)
            rolling_peak60 = pd.Series(close_vals).rolling(window=60, min_periods=1).max().values
            drawdown60 = np.divide(rolling_peak60 - close_vals, rolling_peak60, out=np.zeros_like(close_vals), where=rolling_peak60!=0)
            mkt_pd['mkt_{}_drawdown60'.format(freq)] = drawdown60
            # Regime proxy: SMA20 - SMA60 (raw difference)
            sma60 = pd.Series(close_vals).rolling(window=60, min_periods=1).mean().values
            regime = sma20 - sma60
            mkt_pd['mkt_{}_regime_sma20_60'.format(freq)] = regime
        except Exception as e:
            print(f"Warning: Unable to compute market indicators (SMA/RSI/ATR): {e}")

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

    def _gen_market_feat_from_stock_data(self, freq):
        """
        Generate market features from stock data by aggregating across all stocks.
        This is used when index data file is not available.
        """
        # Load stock data file
        stock_fpath, error_msg = get_stock_data_file(self.config)
        if stock_fpath is None:
            raise ValueError(f"Cannot load stock data file to generate market features. {error_msg}")
        
        # Load stock data
        raw_stock_data = pd.DataFrame(pd.read_csv(stock_fpath, header=0, usecols=['date', 'stock']+list(self.config.use_features)))
        raw_stock_data['date'] = pd.to_datetime(raw_stock_data['date'])
        
        # Aggregate across stocks to get market-level data (average across all stocks per date)
        raw_data = raw_stock_data.groupby(['date'])[self.config.use_features].mean().reset_index()
        raw_data.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
        
        isHasFineData = False  # Stock data is typically daily, not fine frequency
        
        # For market index features (fallback from stock), use mafia_T_w (e.g., 30)
        if hasattr(self.config, 'mafia_T_w') and isinstance(self.config.mafia_T_w, int):
            cur_winsize = self.config.mafia_T_w
        else:
            raise ValueError("Invalid freq[p1]: {}".format(freq))
        
        if not TALIB_AVAILABLE:
            # Simple moving average fallback using pandas
            ma_ay = pd.Series(raw_data['close']).rolling(window=cur_winsize+1, min_periods=1).mean().values
        else:
            ma_func = abstract.Function('ma')
            ma_ay = ma_func(np.array(raw_data['close']), timeperiod=cur_winsize+1)
        temp = {'date': np.array(raw_data['date']), 'mkt_{}_close'.format(freq): np.array(raw_data['close']), 'mkt_{}_ma'.format(freq): ma_ay}
        for change_feat in self.config.use_features:
            feat_vals = np.array(raw_data[change_feat])
            if len(feat_vals) <= 1:
                cg_ay = np.zeros(len(raw_data))
            else:
                prev_vals = feat_vals[:-1]
                cur_vals = feat_vals[1:]
                cg_ay = np.divide(cur_vals, prev_vals, out=np.ones_like(cur_vals, dtype=float), where=prev_vals!=0)
                cg_ay[cg_ay==0] = 1
                cg_ay = cg_ay - 1
                cg_ay = np.append([0], cg_ay, axis=0)
            cg_ay = cg_ay * self.config.feat_scaler
            temp['mkt_{}_{}_w{}'.format(freq, change_feat, 1)] = cg_ay
            for widx in range(2, cur_winsize+1):
                temp['mkt_{}_{}_w{}'.format(freq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
        mkt_pd = pd.DataFrame(temp)
        # Add market technical indicators (SMA20, RSI14, ATR14)
        try:
            close_vals = np.array(raw_data['close'])
            high_vals = np.array(raw_data['high']) if 'high' in raw_data.columns else close_vals
            low_vals = np.array(raw_data['low']) if 'low' in raw_data.columns else close_vals
            sma20 = pd.Series(close_vals).rolling(window=20, min_periods=1).mean().values
            delta = np.diff(close_vals)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(alpha=1.0/14, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1.0/14, adjust=False).mean().values
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain, dtype=float), where=avg_loss!=0)
            rsi14 = 100 - (100 / (1 + rs))
            rsi14 = np.concatenate([[50.0], rsi14])
            if len(rsi14) < len(close_vals):
                rsi14 = np.pad(rsi14, (0, len(close_vals) - len(rsi14)), mode='edge')
            tr = np.zeros(len(close_vals))
            for i in range(1, len(close_vals)):
                tr[i] = max(
                    high_vals[i] - low_vals[i],
                    abs(high_vals[i] - close_vals[i-1]),
                    abs(low_vals[i] - close_vals[i-1])
                )
            atr14 = pd.Series(tr).ewm(alpha=1.0/14, adjust=False).mean().values
            mkt_pd['mkt_{}_sma20'.format(freq)] = sma20
            mkt_pd['mkt_{}_rsi14'.format(freq)] = rsi14
            mkt_pd['mkt_{}_atr14'.format(freq)] = atr14
        except Exception as e:
            print(f"Warning: Unable to compute market indicators (SMA/RSI/ATR): {e}")

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
            # date, stock, close_w1, close_w2, .., open_w1, ..., close/open/high/low/volume_w{1-4}
            for change_feat in self.config.use_features:
                feat_vals = np.array(dataSig[change_feat])
                if len(feat_vals) <= 1:
                    cg_ay = np.zeros(len(dataSig))
                else:
                    prev_vals = feat_vals[:-1]
                    cur_vals = feat_vals[1:]
                    cg_ay = np.divide(cur_vals, prev_vals, out=np.ones_like(cur_vals, dtype=float), where=prev_vals!=0)
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
