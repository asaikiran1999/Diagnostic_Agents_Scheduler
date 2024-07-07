import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

@st.cache(allow_output_mutation=True)
def clustering(x, y):
    df = pd.read_csv("https://raw.githubusercontent.com/asaikiran1999/diagnostic-center-agents-sheduling/main/final_data.csv")
    df1 = df.drop(df.columns[[0, 2, 3, 6, 7, 9, 10, 12, 22, 23, 24, 25]], axis=1)
    df3 = df1[df1['Sample Collection Date'] == x]

    le = LabelEncoder()
    df3['patient location'] = le.fit_transform(df3['patient location'])
    df3['Diagnostic Centers'] = le.fit_transform(df3['Diagnostic Centers'])
    df3['Availabilty time (Patient)'] = le.fit_transform(df3['Availabilty time (Patient)'])
    X = df3[['patient location', 'Diagnostic Centers', 'shortest distance Patient-Pathlab(m)', 'Availabilty time (Patient)']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    X = pd.DataFrame(scaled, columns=['patient location', 'Diagnostic Centers', 'shortest distance Patient-Pathlab(m)', 'Availabilty time (Patient)'])

    kmeans = KMeans(n_clusters=y, random_state=1234)
    kmeans.fit(X)

    labels = kmeans.labels_
    df3["Agent id "] = labels
    df4 = df3.copy()

    return df4

def show_agent_schedule(df, agent_id):
    df5 = df[df['Agent id '] == agent_id]
    df5.index = range(df5.shape[0])

    df5['avail'] = df5['Availabilty time (Patient)'].apply(lambda x: int(x.split('to')[0].strip().split(':')[0]))
    df6 = df5.sort_values(['avail', 'shortest distance Patient-Pathlab(m)'], ascending=[True, True])
    df7 = df6.drop(['avail', 'Test Booking Time HH:MM', 'Test Booking Date', 'shortest distance Patient-Pathlab(m)', 'Sample Collection Date'], axis=1)
    df7 = df7.drop(df7.columns[10], axis=1)
    first_column = df7.pop('Availabilty time (Patient)')
    df7.insert(0, 'Availabilty time (Patient)', first_column)

    return df7.to_numpy().tolist()

def main():
    st.title('Scheduling for Agent')

    # Sidebar inputs
    date = st.text_input('Enter date for schedule generation (YYYY-MM-DD):')
    num_agents = st.number_input('Enter number of agents:', min_value=1, step=1)

    if st.button('Generate Clusters'):
        if date and num_agents:
            df_result = clustering(date, num_agents)
            st.success('Clusters generated successfully!')

    # Display agent schedule
    agent_id = st.number_input('Enter agent ID to show schedule:', min_value=0, step=1, value=0)
    if st.button('Show Agent Schedule'):
        if 'df_result' in locals():
            schedule_data = show_agent_schedule(df_result, agent_id)
            st.json({'success': True, 'data': schedule_data})

if __name__ == '__main__':
    main()
