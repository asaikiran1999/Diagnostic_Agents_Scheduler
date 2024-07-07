import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

@st.cache(allow_output_mutation=True)
def clustering(x, y):
    df = pd.read_csv("https://raw.githubusercontent.com/asaikiran1999/asaikiran1999/Diagnostic_Agents_Scheduler/main/Sampledata.csv")
    df = df.drop(df.columns[[0, 2, 3, 6, 7, 9, 10, 12, 22, 23, 24, 25]], axis=1)
    df_filtered = df[df['Sample Collection Date'] == x].copy()

    le = LabelEncoder()
    df_filtered['patient location'] = le.fit_transform(df_filtered['patient location'])
    df_filtered['Diagnostic Centers'] = le.fit_transform(df_filtered['Diagnostic Centers'])
    df_filtered['Availabilty time (Patient)'] = le.fit_transform(df_filtered['Availabilty time (Patient)'])
    X = df_filtered[['patient location', 'Diagnostic Centers', 'shortest distance Patient-Pathlab(m)', 'Availabilty time (Patient)']].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    kmeans = KMeans(n_clusters=y, random_state=1234)
    kmeans.fit(X_scaled_df)

    labels = kmeans.labels_
    df_filtered["Agent id "] = labels

    return df_filtered.copy()

def show_agent_schedule(df, agent_id):
    df_agent = df[df['Agent id '] == agent_id].copy()
    df_agent['avail'] = df_agent['Availabilty time (Patient)'].apply(lambda x: int(x.split('to')[0].strip().split(':')[0]))
    df_agent_sorted = df_agent.sort_values(['avail', 'shortest distance Patient-Pathlab(m)'], ascending=[True, True])
    df_agent_filtered = df_agent_sorted.drop(['avail', 'Test Booking Time HH:MM', 'Test Booking Date', 'shortest distance Patient-Pathlab(m)', 'Sample Collection Date'], axis=1)
    df_agent_filtered = df_agent_filtered.drop(df_agent_filtered.columns[10], axis=1)
    first_column = df_agent_filtered.pop('Availabilty time (Patient)')
    df_agent_filtered.insert(0, 'Availabilty time (Patient)', first_column)

    return df_agent_filtered.to_numpy().tolist()

def main():
    st.title('Scheduling for Agent')

    # Sidebar inputs
    if st.button('Generate Clusters'):
        date = st.text_input('Enter date for schedule generation (YYYY-MM-DD):')
        num_agents = st.number_input('Enter number of agents:', min_value=1, step=1)

        if date and num_agents:
            df_result = clustering(date, num_agents)
            st.success('Clusters generated successfully!')

    # Display agent schedule
    if 'df_result' in st.session_state:
        agent_id = st.number_input('Enter agent ID to show schedule:', min_value=0, step=1, value=0)
        if st.button('Show Agent Schedule'):
            schedule_data = show_agent_schedule(st.session_state.df_result, agent_id)
            st.json({'success': True, 'data': schedule_data})

if __name__ == '__main__':
    main()
