import pandas as pd
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


sns.set_palette(np.array(sns.color_palette('Paired'))[[0, 1, 4, 5]])


def load_data(filename='spaceRL_test.csv'):
    return pd.read_csv(filename)


def save_to_matlab_array():
    keep = set()

    ntrial = 240

    df = load_data()
    ids = df['prolificID'].unique()
    # df = df[df['prolificID'] == ids[0]]

    for i in ids:
        if len(df[df['prolificID'] == i]) == ntrial:
            keep.add(i)

    df = df[df['prolificID'].isin(keep)]

    con = np.zeros((len(keep), ntrial))
    cho = np.zeros((len(keep), ntrial))
    corr = np.zeros((len(keep),  ntrial))
    out = np.zeros((len(keep), ntrial))
    cfout = np.zeros((len(keep), ntrial))
    p1 = np.zeros((len(keep), ntrial))
    p2 = np.zeros((len(keep), ntrial))

    for i, sess in ((0, 1), (1, 2)):
        k = [np.arange(120), np.arange(120, 240)]
        for j, sub in enumerate(keep):
            p1[j, k[i]] = df[(df['session'] == sess)
                            & (df['prolificID'] == sub)
                                ].sort_values('t')['p1']
            p2[j, k[i]] = df[(df['session'] == sess)
                    & (df['prolificID'] == sub)
                    ].sort_values('t')['p2']
            corr[j, k[i]] = df[(df['session'] == sess)
                    & (df['prolificID'] == sub)
                    ].sort_values('t')['corr']
            con[j, k[i]] = df[(df['session'] == sess)
                    & (df['prolificID'] == sub)
                    ].sort_values('t')['con']
            cho[j, k[i]] = df[(df['session'] == sess)
                    & (df['prolificID'] == sub)
                    ].sort_values('t')['choice']
            out[j, k[i]] = df[(df['session'] == sess)
                              & (df['prolificID'] == sub)
                              ].sort_values('t')['outcome']
            cfout[j, k[i]] = df[(df['session'] == sess)
                              & (df['prolificID'] == sub)
                              ].sort_values('t')['cfoutcome']

    scipy.io.savemat('spaceRL.mat',
                     dict(
                         out=out,
                         cfout=cfout,
                         cho=cho,
                         con=con,
                         corr=corr,
                         p1=p1,
                         p2=p2))


def corr_by_con(df, keep):
    y = np.zeros((len(keep), 4, 30), dtype=int)
    for i, sub in enumerate(keep):
        for j, con in enumerate(range(1, 5)):
            y[i, j, :] = df[  # (df['session'] == sess)
                (df['con'] == con)
                & (df['prolificID'] == sub)
                ].sort_values('t')['corr']

    for i in range(4):
        x = np.ones((len(keep), 60), dtype=int) * np.arange(1, 61)
        dd = pd.DataFrame(
            {'corr': y[:, i, :].flatten(), 't': x.flatten()})
        sns.lineplot(
            x='t', y='corr',
            label=f'Condition {i + 1}',
            data=dd, ci='sem', color=f'C{i}')
        plt.ylim([0, 1])
        plt.show()


def main():
    # save_to_matlab_array()

    #
    df = load_data()
    ids = df['prolificID'].unique()

    keep = set()
    for i in ids:
        if len(df[df['prolificID'] == i]) == 240:
            keep.add(i)

    n = len(keep)
    print(f'N={n}')

    df = df[df['prolificID'].isin(keep)]
    # df['t'] = df['session'] * (df['t']+1)
    fig, ax = plt.subplots(2, 3)
    patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i, t in
               enumerate(['partial', 'complete', 'partial', 'complete'])]
    ax[0, 0].legend(handles=patches, loc='upper left')

    for sess in (1, 2):
        df_sess = df[df['session']==sess]

        # corr_by_con(df, keep)

        # Compute median + mean by sub and cond
        # --------------------------------------------------------------------------------------------------------- #

        med = df_sess.groupby(
            ['prolificID', 'con'], as_index=False
        ).median()
        avg = df_sess.groupby(
            ['prolificID', 'con'], as_index=False
        ).mean()

        # Plots
        # -------------------------------------------------------------------------------------------------------- #

        sns.barplot(x='con', y='rt',
                    data=med,
                    ci=68, alpha=.5, zorder=0,
                    error_kw=dict(zorder=2), ax=ax[sess-1, 0])
        sns.stripplot(x='con', y='rt',
                      data=med,
                      edgecolor='w', linewidth=1.2, size=5, zorder=1, ax=ax[sess-1, 0])

        # -------------------------------------------------------------------------------------------------------- #

        ax[sess-1, 1].set_title(f'Sess. {sess}')
        sns.barplot(x='con', y='fireCount',
                    data=avg,
                    ci=68, alpha=.5, zorder=0,
                    error_kw=dict(zorder=2), ax=ax[sess-1, 1])
        sns.stripplot(x='con', y='fireCount',
                      data=avg,
                      edgecolor='w', linewidth=1.2, size=5, zorder=1, ax=ax[sess-1, 1])

        # -------------------------------------------------------------------------------------------------------- #

        avg_move = avg.loc[:, ['upCount', 'downCount', 'leftCount', 'rightCount']].sum(axis=1)
        avg['moveCount'] = avg_move

        sns.barplot(x='con', y='moveCount',
                    data=avg,
                    ci=68, alpha=.5, zorder=0,
                    error_kw=dict(zorder=2), ax=ax[sess-1, 2])
        sns.stripplot(x='con', y='moveCount',
                      data=avg,
                      edgecolor='w', linewidth=1.2, size=5, zorder=1, ax=ax[sess-1, 2])

        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
