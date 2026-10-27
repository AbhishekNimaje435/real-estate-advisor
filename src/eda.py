"""Generate EDA figures for the key questions in the project brief."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config as C  # noqa: E402
from src.data_utils import add_features, clean, load_data  # noqa: E402


def save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def main():
    df = add_features(clean(load_data()))
    sns.set_theme(style="whitegrid")

    # Q1: price distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax)
    ax.set_title("Q1: Distribution of property prices (Lakhs)")
    save(fig, "q01_price_distribution")

    # Q2: size distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax)
    ax.set_title("Q2: Distribution of property size (sq ft)")
    save(fig, "q02_size_distribution")

    # Q3: price per sq ft by property type
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", ax=ax)
    ax.set_title("Q3: Price per sq ft by property type")
    save(fig, "q03_ppsqft_by_type")

    # Q4: size vs price
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=C.RANDOM_STATE),
                    x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3, ax=ax)
    ax.set_title("Q4: Size vs price")
    save(fig, "q04_size_vs_price")

    # Q6: average price per sq ft by state
    fig, ax = plt.subplots(figsize=(10, 5))
    by_state = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
    by_state.plot(kind="barh", ax=ax)
    ax.set_title("Q6: Average price per sq ft by state")
    save(fig, "q06_ppsqft_by_state")

    # Q11: correlation heatmap
    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Q11: Correlation between numeric features")
    save(fig, "q11_correlation")

    # Q14: price by furnished status
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax)
    ax.set_title("Q14: Price by furnished status")
    save(fig, "q14_price_by_furnishing")

    # Q16: owner type counts
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Owner_Type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Q16: Properties by owner type")
    save(fig, "q16_owner_type")

    # Q20: transport accessibility vs price per sq ft
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", ax=ax)
    ax.set_title("Q20: Transport accessibility vs price per sq ft")
    save(fig, "q20_transport_vs_ppsqft")

    print(f"EDA figures saved in {C.FIGURES_DIR}")
    print(df[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].describe().round(2))


if __name__ == "__main__":
    main()
