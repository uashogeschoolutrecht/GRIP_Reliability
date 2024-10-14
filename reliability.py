# %%
#
import pandas as pd
import pingouin as pg

data = pd.read_excel('Results/results_per_day.xlsx')
data = data.dropna(axis=1)
data = data.sort_values(by=['day'])
subject_counts = data['subject'].value_counts()
results = {}
final_df = pd.DataFrame()

# Minimum duration of 10 hours
data = data.loc[data['Epochs_of_1minute'] >= 600]

for i in range(1, 8):
    subjects = subject_counts.loc[subject_counts >= i * 2].index
    subjects_data = data.loc[data['subject'].isin(subjects)]

    # Get the first 'i' rows for each subject
    subjects_data = subjects_data.set_index('subject')
    subjects_data = subjects_data.loc[:, 'average_activity_level':]

    # Get the first 'i' rows for each subject and calculate the mean
    first = subjects_data.groupby('subject').head(i)
    mean_values_first = first.groupby('subject').mean()
    # Add a column to indicate 'first' group
    mean_values_first['group_label'] = 1

    # Get the second 'i' rows for each subject (from i to i+7)
    second = subjects_data.groupby('subject').apply(lambda x: x.iloc[i:i+7])
    mean_values_second = second.groupby('subject').mean()
    # Add a column to indicate 'second' group
    mean_values_second['group_label'] = 2

    # Concatenate the mean values of the first and second groups
    final_mean_df = pd.concat(
        [mean_values_first, mean_values_second]).reset_index()
    print(f'Aantal dagen: {i}, aantal proefpersonen {len(final_mean_df) / 2}')
    for variable in subjects_data.columns:
        try:
            icc = pg.intraclass_corr(data=final_mean_df, targets='subject', raters='group_label',
                                     ratings=variable).round(3)
            icc2 = icc.loc[icc['Type'] == 'ICC2']
            results[f'ICC_{variable}'] = icc2['ICC'].values[0]
            # results[f'CI_days_{i}_var_{variable}'] = icc2['CI95%'].values[0]
        except:
            continue
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=[i])
    if final_df.empty:
        final_df = results_df
    else:
        final_df = pd.concat((final_df, results_df), axis=1)
final_df.to_excel('Results/ICC_waardes.xlsx')

# %%
