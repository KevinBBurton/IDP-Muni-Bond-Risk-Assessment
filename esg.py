import pandas


def get_NRI(county, state):
    """
    Function that returns a National Risk score and rating for a given US county and state using the NRI table
    :param county: County
    :param state: State
    :return: Score and rating or None, if they could not be retrieved
    """
    df = pandas.read_csv('NRI_Table_Counties.csv', delimiter=',')
    df = df[['STATE', 'COUNTY', 'RISK_SCORE', 'RISK_RATNG']]

    try:
        df = df[df['STATE'] == state]
    except KeyError:
        print("State could not be found in NRI records.")

    df.set_index('COUNTY', inplace=True)
    try:
        score = df.at[county, 'RISK_SCORE']
        rating = df.at[county, 'RISK_RATNG']
    except KeyError:
        print("County could not be found in NRI records for the given state.")
        score = None
        rating = None

    return score, rating
