import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Simple Business Dashboard", layout="wide", initial_sidebar_state="expanded")

#  Data Loading and Basic Preparation 
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("Error: data.csv not found. Please place it in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return pd.DataFrame()

    #  Essential Feature Engineering 
    if 'order_purchase_timestamp' in df.columns:
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
        df.dropna(subset=['order_purchase_timestamp'], inplace=True)
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_month'] = df['order_purchase_timestamp'].dt.month
        # Create Year-Month for correct sorting/grouping across years
        df['order_year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

    else:
        st.error("Critical column 'order_purchase_timestamp' not found. Cannot create filters or trends.")
        return pd.DataFrame() # If data is missing

    # Convert payment value to numeric
    if 'payment_value' in df.columns:
        df['payment_value'] = pd.to_numeric(df['payment_value'], errors='coerce').fillna(0)

    if 'payment_status' in df.columns:
         df['is_paid'] = df['payment_status'] == 'paid'
    elif 'payment_value' in df.columns:
        df['is_paid'] = df['payment_value'] > 0
        st.warning("Column 'payment_status' not found. Assuming orders with payment_value > 0 are 'paid'.")
    else:
         df['is_paid'] = False 

    if 'order_status' in df.columns:
        df['is_canceled'] = df['order_status'] == 'canceled'
    else:
        st.warning("Column 'order_status' not found. Cannot calculate cancellation rates accurately.")
        df['is_canceled'] = False

    if 'customer_state' in df.columns:
        df['customer_state'] = df['customer_state'].astype(str)

    return df

df = load_data()

#  Sidebar Filters 
st.sidebar.header("Filter Data")

if not df.empty and all(col in df.columns for col in ['order_year', 'order_month', 'customer_state']):
    # Year Filter
    years = sorted(df['order_year'].unique().astype(int))
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)

    # Month Filter
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_dict = {i: name for i, name in enumerate(month_names, 1)}
    selected_months = st.sidebar.multiselect("Select Months", months, default=months, format_func=lambda x: month_dict[x])

    # State Filter
    states = sorted(df['customer_state'].unique().tolist())
    selected_states = st.sidebar.multiselect("Select States", states, default=states)

    # Apply filters
    df_filtered = df[
        (df['order_year'].isin(selected_years if selected_years else years)) &
        (df['order_month'].isin(selected_months if selected_months else months)) &
        (df['customer_state'].isin(selected_states if selected_states else states))
    ].copy()

else:
    st.sidebar.warning("Data is empty or essential columns for filtering are missing.")
    df_filtered = pd.DataFrame()

#  Main Dashboard 
st.title("Simple Business Performance Dashboard")

if df_filtered.empty:
    st.warning("No data matches the selected filters or initial data is missing.")
else:
    #  Key Metrics 
    st.header("Key Performance Indicators")

    # Calculate metrics based on paid orders within the filtered data
    paid_orders_df = df_filtered[df_filtered['is_paid']].copy()
    total_revenue = paid_orders_df['payment_value'].sum()
    total_paid_orders = paid_orders_df['order_id'].nunique() # Count unique paid orders
    avg_order_value = total_revenue / total_paid_orders if total_paid_orders > 0 else 0 # avoiding div by 0
    
    customer_order_counts = df_filtered.groupby('customer_id')['order_id'].nunique()
    returning_customers = customer_order_counts[customer_order_counts > 1].count()
    total_customers = customer_order_counts.count()
    retention_rate = (returning_customers / total_customers) * 100
    one_time_customers = customer_order_counts[customer_order_counts == 1].count()
    churn_rate = (one_time_customers / total_customers) * 100
    top_category = df.groupby('category')['order_value'].sum().idxmax()
    top_category_value = df.groupby('category')['order_value'].sum().max()


    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Paid Orders", f"{total_paid_orders:,}")
    col3.metric("Average Order Value", f"${avg_order_value:,.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customer Retention Rate", f"{retention_rate:,.2f}%")
    col2.metric("Total Churn Rate", f"{churn_rate:,}%")
    col3.metric("Top Product Category", f"{top_category}")

    st.markdown("---") # Separator

    #  Sales Trend 
    st.header("Monthly Trends")
    col1, col2 = st.columns(2)
    # Prepare data for plot (Revenue by Year-Month)
    with col1:
        col1.markdown("Month vs Revenue")
        if not paid_orders_df.empty and 'order_year_month' in paid_orders_df.columns:
            monthly_revenue = paid_orders_df.groupby('order_year_month')['payment_value'].sum().reset_index()
            monthly_revenue.sort_values('order_year_month', inplace=True) # Ensure sorted by date

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=monthly_revenue, x='order_year_month', y='payment_value', marker='o', ax=ax)
            ax.set_title('Total Revenue per Month')
            ax.set_xlabel('Month (YYYY-MM)')
            ax.set_ylabel('Total Revenue ($)')
            plt.xticks(rotation=45)
            fig.tight_layout()
            col1.pyplot(fig)
        else:
            col1.info("No paid order data available for the selected period to show monthly trend.")
    with col2:
        col2.markdown("Month vs Orders Count")
        if not paid_orders_df.empty and 'order_year_month' in paid_orders_df.columns:
            monthly_orders = paid_orders_df.groupby('order_year_month')['order_id'].count().reset_index()
            monthly_orders.sort_values('order_year_month', inplace=True) # Ensure sorted by date

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=monthly_orders,x='order_year_month',y="order_id", marker='o', ax=ax)
            ax.set_title('Total orders per Month')
            ax.set_xlabel('Month (YYYY-MM)')
            ax.set_ylabel('Orders Count')
            plt.xticks(rotation=45)
            fig.tight_layout()
            col2.pyplot(fig)
        else:
            col2.info("No paid order data available for the selected period to show monthly trend.")

    st.markdown("---")
    st.header("Order Trends")
    col1, col2 = st.columns(2)

    with col1:
        if 'order_status' in df_filtered.columns:
            orders_per_customer = df_filtered.groupby('customer_id')['order_id'].nunique()
            fig, ax = plt.subplots(figsize=(8, 4))
            orders_per_customer.hist(bins=20, ax=ax)
            ax.set_title('Distribution of Orders per Customer')
            ax.set_xlabel('Number of Orders')
            ax.set_ylabel('Number of Customers')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            col2.info("Order status information not available.")

    with col2:
        if 'order_status' in df_filtered.columns:
            status_counts = df_filtered['order_status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']

            fig_status, ax_status = plt.subplots(figsize=(8,4))
            sns.barplot(data=status_counts, x='status', y='count', ax=ax_status)
            ax_status.set_title('Order Status Distribution')
            ax_status.set_xlabel('Order Status')
            ax_status.set_ylabel('Number of Orders')
            plt.xticks(rotation=45)
            plt.yscale("log")
            fig_status.tight_layout()
            col2.pyplot(fig_status)
        else:
            col2.info("Order status information not available.")

    st.markdown("---")
    st.header("Product Performance Understanding")
    st.markdown("Analyzing which products sell best and identifying underperformers (based on paid orders).")

    # Check required columns
    required_cols = {'product_id', 'category', 'quantity'}
    if required_cols.issubset(paid_orders_df.columns):
        product_sales = paid_orders_df.groupby(['product_id', 'category']).agg(
            total_revenue=('payment_value', 'sum'),
            total_quantity=('quantity', 'sum'),
            order_count=('order_id', 'nunique')
        ).reset_index()
    else:
        product_sales = pd.DataFrame()
        st.warning("Missing one or more columns: product_id, category, or quantity.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Revenue by Product Category")
        if 'category' in paid_orders_df.columns and not paid_orders_df.empty:
            category_revenue = (
                paid_orders_df.groupby('category')['payment_value']
                .sum()
                .reset_index()
                .sort_values('payment_value', ascending=False)
            )
            if not category_revenue.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=category_revenue.head(10), x='category', y='payment_value', ax=ax, palette='Blues')
                ax.set_title("Top 10 Categories by Revenue")
                ax.set_xlabel("Category")
                ax.set_ylabel("Total Revenue ($)")
                ax.tick_params(axis='x', rotation=45)
                plt.yscale("log")
                st.pyplot(fig)
            else:
                st.info("No category revenue data available.")
        else:
            st.info("Paid order data or category column is missing.")


    with col2:
        tab1, tab2 = st.tabs(["Top Products", "Low Performers"])

        with tab1:
            st.subheader("Top Revenue Generating Products")
            if not product_sales.empty:
                top_products = product_sales.sort_values('total_revenue', ascending=False)
                st.dataframe(
                    top_products[['product_id', 'category', 'total_revenue', 'total_quantity']].head(10)
                    .style.format({'total_revenue': '${:,.2f}', 'total_quantity': '{:,}'}),
                    height=400
                )
            else:
                st.info("No product sales data available.")

        with tab2:
            st.subheader("Low Performing Products")
            if not product_sales.empty and product_sales['total_revenue'].sum() > 0:
                threshold = product_sales['total_revenue'].quantile(0.10)
                low_performers = product_sales[
                    (product_sales['total_revenue'] <= threshold) &
                    (product_sales['total_revenue'] > 0)
                ].sort_values('total_revenue')

                if not low_performers.empty:
                    st.write(f"Products in bottom 10% by revenue (Threshold: ${threshold:,.2f}):")
                    st.dataframe(
                        low_performers[['product_id', 'category', 'total_revenue', 'total_quantity']].head(10)
                        .style.format({'total_revenue': '${:,.2f}', 'total_quantity': '{:,}'}),
                        height=400
                    )
                else:
                    st.info("No low-performing products with revenue above $0.")
            elif product_sales.empty:
                st.info("No product revenue data.")
            else:
                st.info("Total revenue is zero. Cannot identify low performers.")

    st.markdown("---")



    st.header("Region Trends")
    col1, col2 = st.columns(2)

    with col1:
        col1.markdown("Top Revenue Generating Regions (States)")
        if not paid_orders_df.empty and 'customer_state' in paid_orders_df.columns:
            state_revenue = paid_orders_df.groupby('customer_state')['payment_value'].sum().reset_index().sort_values('payment_value', ascending=False)

            if not state_revenue.empty:
                fig, ax = plt.subplots(figsize=(15, 7))
                top_n = min(15, len(state_revenue))
                sns.barplot(data=state_revenue.head(top_n), x='customer_state', y='payment_value', ax=ax)
                ax.set_title(f'Top {top_n} States by Revenue')
                ax.set_xlabel('State')
                ax.set_ylabel('Total Revenue ($)')
                ax.tick_params(axis='x', rotation=45)
                col1.pyplot(fig)
                col1.markdown("Top Revenue Generating Regions (States) in Order")
                col1.dataframe(state_revenue.reset_index())
            else:
                 st.info("No revenue data found for states in the filtered data.")

    with col2:
        col2.markdown("Customer Distribution vs Regions")
        if not df_filtered.empty:
            customers_by_state = df_filtered.groupby('customer_state')['customer_id'].nunique().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(15, 8))
            customers_by_state.plot(kind='bar', ax=ax)
            ax.set_title('Customer Distribution by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Number of Unique Customers')
            plt.xticks(rotation=45)
            plt.tight_layout()
            col2.pyplot(fig)
            col2.markdown("Order Status vs Region (in %)")
            status_counts = df.groupby(['customer_state', 'order_status'])['order_id'].count().reset_index()
            total_orders_per_state = df.groupby('customer_state')['order_id'].count().reset_index()
            total_orders_per_state.rename(columns={'order_id': 'total_orders'}, inplace=True)
            merged = pd.merge(status_counts, total_orders_per_state, on='customer_state')
            merged['percentage'] = (merged['order_id'] / merged['total_orders']) * 100
            pivot_table = merged.pivot(index='customer_state', columns='order_status', values='percentage').fillna(0)   
            col2.dataframe(pivot_table)
        else:
            col2.info("No revenue data for states in the filtered data.")

    #  Data Preview 
    st.markdown("---")

    st.header("Payment and Order Trends")

    col_pay1, col_pay2 = st.columns(2)

    with col_pay1:
        st.subheader("Payment Method Popularity")
        st.markdown("Distribution of payment methods used (based on all filtered orders).")
        if not df_filtered.empty and 'payment_type' in df_filtered.columns:
            payment_share_count = df_filtered['payment_type'].value_counts()
            payment_share_percent = df_filtered['payment_type'].value_counts(normalize=True) * 100

            if not payment_share_count.empty:
                top_method = payment_share_count.index[0]
                top_method_perc = payment_share_percent.iloc[0]
                st.metric(f"Most Common Method", f"{top_method}", f"{top_method_perc:.1f}% of Orders")

                fig, ax = plt.subplots(figsize=(10, 6))
                # Using muted categorical palette for counts
                sns.barplot(x=payment_share_count.index, y=payment_share_count.values, ax=ax)
                ax.set_ylabel('Number of Orders')
                ax.set_xlabel('Payment Type')
                ax.set_title('Payment Method Share (by Order Count)')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.info("No payment type data available.")
        else:
            st.info("Payment type data missing.")

    with col_pay2:
        st.subheader("Payment Type Correlations")
        st.markdown("Comparing Average Order Value (AOV) and Cancellation Rates across payment methods.")

        tab_corr1, tab_corr2 = st.tabs(["AOV by Payment Type", "Cancellation Rate by Payment Type"])

        with tab_corr1:
             st.markdown("**Average Order Value (AOV) Comparison**")
             if not paid_orders_df.empty and 'payment_type' in paid_orders_df.columns:
                 aov_by_payment = paid_orders_df.groupby('payment_type')['payment_value'].mean().reset_index().sort_values('payment_value', ascending=False)
                 if not aov_by_payment.empty:
                     fig, ax = plt.subplots(figsize=(8, 5))
                     # Using sequential green palette for AOV intensity
                     sns.barplot(data=aov_by_payment, x='payment_value', y='payment_type', ax=ax, orient='h')
                     ax.set_title("Average Order Value (Paid Orders) by Payment Type")
                     ax.set_xlabel("Average Order Value ($)")
                     ax.set_ylabel("Payment Type")
                     st.pyplot(fig)
                 else:
                      st.info("Could not calculate AOV by payment type.")
             else:
                 st.info("Payment type or paid order data missing.")

        with tab_corr2:
            st.markdown("**Cancellation Rate Comparison**")
            if not df_filtered.empty and 'payment_type' in df_filtered.columns and 'is_canceled' in df_filtered.columns and 'order_id' in df_filtered.columns:
                 cancellation_by_payment = df_filtered.groupby('payment_type').agg(
                     total_orders=('order_id', 'nunique'),
                     canceled_orders=('is_canceled', 'sum')
                 ).reset_index()

                 cancellation_by_payment['cancellation_rate'] = np.where(
                     cancellation_by_payment['total_orders'] > 0,
                     (cancellation_by_payment['canceled_orders'] / cancellation_by_payment['total_orders'] * 100),
                     0
                 )

                 if not cancellation_by_payment.empty:
                      fig, ax = plt.subplots(figsize=(8, 4))
                      sns.barplot(data=cancellation_by_payment.sort_values('cancellation_rate', ascending=False), x='cancellation_rate', y='payment_type', ax=ax, orient='h')
                      ax.set_title("Cancellation Rate (%) by Payment Type")
                      ax.set_xlabel("Cancellation Rate (%)")
                      ax.set_ylabel("Payment Type")
                      st.pyplot(fig)
                 else:
                     st.info("No cancellation data available by payment type.")
            else:
                st.info("Payment type, cancellation status, or order ID data missing.")


    st.markdown("---")
    with st.expander("Show Filtered Data Preview"):
        st.dataframe(df_filtered.head(50))