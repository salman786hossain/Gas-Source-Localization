import pandas as pd
from datetime import timedelta



class time:
    start_time = 0
    oldtime = 0

class syn:
    t = time()

    def syn_time(self, df1, df2):
        self.t.oldtime = df1.index[0]
        self.t.start_time = df1.index[0]


        for i in range(len(df2)-1):
            dt = df2['Flight time'][i]
            wind1 = df2['Wind Speed'][i]
            wind2 = df2['Wind Direction'][i]
            # print(df2['Flight time'][i])

            minutes = int(dt[:2])
            seconds = int(dt[3:-1])
            newtime = self.t.start_time + pd.Timedelta(minutes=minutes, seconds=seconds)
            # print (newtime)
            nearst = df1.index.searchsorted(newtime)
            # print(nearst,newtime)
             # Adjust the index if it's out of bounds
            if nearst >= len(df1.index):
                nearst = len(df1.index) - 1
            mask = (df1.index < df1.index[nearst]) & (df1.index >= self.t.oldtime)
            # print(mask)

            df1.loc[mask, 'wind_speed'] = wind1
            df1.loc[mask,'wind_direction'] = wind2


            self.t.oldtime = newtime
            # print(df1)
            # print(wind)

            checktime1 = df1.index[nearst-1]
            # print(checktime1)
            checktime2 = df1.index[nearst]
            # print(checktime2)
            checkdt =checktime2 - checktime1
            # print(checkdt)
            # break
            interval = timedelta(minutes=1)
            if checkdt > interval:
                self.t.start_time = checktime2

        lastvalue1 = df2['Wind Speed'][len(df2)-1]
        lastvalue2 = df2['Wind Direction'][len(df2) - 1]
        # print(lastvalue)
        df1['wind_speed'] = df1['wind_speed'].fillna(lastvalue1)
        df1['wind_direction'] = df1['wind_direction'].fillna(lastvalue2)


        return df1




if __name__ == '__main__':
    # Load CO2 data into a pandas data frame
    df1 = pd.read_csv("Raw_data/Day_1/Raw_sensordata_day1/Datalog_day1_raw_100m.csv")
    # print(df1)


    df2 = pd.read_csv("Raw_data/Day_1/wind_data_day1/wind_100m_raw_day1.csv")
    # print(df2)

    # Remove all occurrences of UTC from the Time column
    df1['TimeStamp (UTC)'] = df1['TimeStamp (UTC)'].str.replace('UTC', '')
    df1['TimeStamp (UTC)'] = df1['TimeStamp (UTC)'].str.replace('5 9', '')

    # Convert the TimeStamp (UTC) column to datetime format
    df1['TimeStamp (UTC)'] = df1['TimeStamp (UTC)'].apply(lambda x: pd.to_datetime(x.strip(), format='%y%m%d %H%M%S'))
    df1 = df1.set_index('TimeStamp (UTC)')


    # Load wind data into a pandas data frame

    st = syn()
    # print(st)
    syn_df = st.syn_time(df1, df2)
    syn_df.to_csv('QGIS_upsampling_100m_day1.csv')


    # print(syn_df)






