import os
import camelot
import pandas as pd

# Importing the data from pdf file in raw format
file_path=os.path.join(os.getcwd(), 'literature\data_paper.pdf') # This is the file path of data paper
tables=camelot.read_pdf(file_path,pages='4-16', flavor='stream')

# Extracting and pre-processing Table 1 from Page 4
table_1=tables[0].df
table_1=table_1.drop(table_1.columns[4], axis=1)
table_1=table_1.drop(range(8), axis=0)
table_1.columns=['Concrete mixture', 'Sample size, φ×h(mmxmm)',
'Density, ρ(kg/m3)_mean','Density, ρ(kg/m3)_SD', 
'Moisture content, wc(%)_mean','Moisture content, wc(%)_SD',
'P-wave velocity, Vp(m/s)_mean','P-wave velocity, Vp(m/s)_SD',
'Dynamic elastic modulus, Yd(GPa)_mean','Dynamic elastic modulus, Yd(GPa)_SD']
table_1.iloc[0:4,0]='F'
table_1.iloc[5:8,0]='M'
table_1.iloc[9:12,0]='C'
table_1

# Extracting and pre-processing Table 2 from Page 4
table_2=tables[1].df
table_2=table_2.drop(range(9), axis=0)
table_2.columns=['Concrete mixture', 'Sample size, φ×h(mmxmm)',
'Global autocorrelation length, ξg(mm)_mean','Global autocorrelation length, ξg(mm)_SD', 
'Integral range, Xog(mm)_mean','Integral range, Xog(mm)_SD',
'Autocorrelation length of the pore structure, ξp(μm)_mean','Autocorrelation length of the pore structure, ξp(μm)_SD',
'Integral range, Xop(mm)_mean','Integral range, Xop(mm)_SD',
'Mean pore diameter, dp(mm)_mean','Mean pore diameter, dp(mm)_SD',
'Maximum pore diameter, dpmax(mm)_mean','Maximum pore diameter, dpmax(mm)_SD',
'Porosity, po(%)_mean','Porosity, po(%)_SD']
table_2.iloc[0:4,0]='F'
table_2.iloc[5:8,0]='M'
table_2.iloc[9:12,0]='C'

table_2


# Extracting and pre-processing of Table 3 to Table 14 from Pages 5 to 16
tables_list=[]
for tbl_num in range(2,tables.n):
    tbl=tables[tbl_num].df
    tbl=tbl.drop([tbl.columns[0],tbl.columns[6]], axis=1)
    if tbl_num == 2:
        tbl=tbl.drop(range(3), axis=0)
    elif tbl_num == 3:
        tbl=tbl.drop(range(6), axis=0)
    elif tbl_num == 9:
        tbl=tbl.drop(range(5), axis=0)
    elif tbl_num == 10:
        tbl=tbl.drop(range(6), axis=0)
    elif tbl_num == 11:
        tbl=tbl.drop(range(5), axis=0)
    elif tbl_num == 12:
        tbl=tbl.drop(range(7), axis=0)
    else:
        tbl=tbl.drop(range(8), axis=0)
        # print(tab)


    # print(f'Table #{tbl_num+1}')
    # print(tbl.shape,tbl.head(1).iloc[0,0],'_,',tbl.tail(1).iloc[0,0])
    tbl.columns= column_names=['Sample ID', 'Diameter, φ(mm)', 'Height, h(mm)', 'Mass of sample, (kg)', 'Density, ρ(kg/m3)', 'Peak load, Fmax(kN)', 'Compressive failure strength, CS(MPa)']
    tbl['Concrete mixture']=tbl['Sample ID'].str[0]
    tables_list.append(tbl)

tables3_14=pd.concat(tables_list).drop_duplicates()

tables3_14.loc[tables3_14['Sample ID'].str.contains('F4'), 'Sample size, φ×h(mmxmm)'] = '40 × 80'
tables3_14.loc[tables3_14['Sample ID'].str.contains('F7'), 'Sample size, φ×h(mmxmm)'] = '70 × 140'
tables3_14.loc[tables3_14['Sample ID'].str.contains('F11'), 'Sample size, φ×h(mmxmm)'] = '110 × 220'
tables3_14.loc[tables3_14['Sample ID'].str.contains('F16'), 'Sample size, φ×h(mmxmm)'] = '160 × 320'

tables3_14.loc[tables3_14['Sample ID'].str.contains('M4'), 'Sample size, φ×h(mmxmm)'] = '40 × 80'
tables3_14.loc[tables3_14['Sample ID'].str.contains('M7'), 'Sample size, φ×h(mmxmm)'] = '70 × 140'
tables3_14.loc[tables3_14['Sample ID'].str.contains('M11'), 'Sample size, φ×h(mmxmm)'] = '110 × 220'
tables3_14.loc[tables3_14['Sample ID'].str.contains('M16'), 'Sample size, φ×h(mmxmm)'] = '160 × 320'

tables3_14.loc[tables3_14['Sample ID'].str.contains('C4'), 'Sample size, φ×h(mmxmm)'] = '40 × 80'
tables3_14.loc[tables3_14['Sample ID'].str.contains('C7'), 'Sample size, φ×h(mmxmm)'] = '70 × 140'
tables3_14.loc[tables3_14['Sample ID'].str.contains('C11'), 'Sample size, φ×h(mmxmm)'] = '110 × 220'
tables3_14.loc[tables3_14['Sample ID'].str.contains('C16'), 'Sample size, φ×h(mmxmm)'] = '160 × 320'


### Preparing the final data frame
df_final=pd.merge(table_1,table_2)
df_final=pd.merge(df_final,tables3_14)

# removing the un_necessary columns
columns_to_drop=[ 'Sample size, φ×h(mmxmm)','Sample ID','Density, ρ(kg/m3)_mean','Density, ρ(kg/m3)_SD']
df_final=df_final.drop(columns=columns_to_drop)

# Apply pd.to_numeric to all columns except 'Concrete mixture'
df_final = df_final.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name != 'Concrete mixture' else x)

df_final.at[226, 'Density, ρ(kg/m3)'] = 2444.6


df_final.shape

df_final.columns
df_final['Density, ρ(kg/m3)'][df_final['Density, ρ(kg/m3)'].isnull()]


df_final.describe(include='all').T.to_csv(os.path.join(os.getcwd(),'results/dataset2_statSumm.csv'), index=False)



# Downloading data in .csv format
df_final.to_csv(os.path.join(os.getcwd(),'data/dataset2.csv'), index=False)


# df_final.head(3)%>%View()









