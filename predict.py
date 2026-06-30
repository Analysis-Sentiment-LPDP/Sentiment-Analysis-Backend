import logging
from src.inference import SentimentPredictor

# Menonaktifkan log INFO agar tampilan lebih bersih
logging.basicConfig(level=logging.ERROR)


def print_header():
    print("=" * 60)
    print("           LPDP SENTIMENT ANALYSIS")
    print("=" * 60)


def print_result(result):
    print("\n" + "-" * 50)

    print("📝 Original Text")
    print(result["text_original"])

    print("\n🧹 Clean Text")
    print(result["text_preprocessed"])

    sentiment = result["sentiment"]

    if sentiment == "Positif":
        icon = "🟢"
    elif sentiment == "Negatif":
        icon = "🔴"
    else:
        icon = "🟡"

    print("\n🎯 Prediction")
    print(f"{icon} {sentiment}")

    print("\n📊 Confidence")
    print(f"{result['confidence'] * 100:.2f}%")

    print("\n📈 Probabilities")

    probs = result["probabilities"]

    for label in sorted(probs.keys()):
        print(f"{label:<10}: {probs[label] * 100:.2f}%")

    print("-" * 50)


def main():

    predictor = SentimentPredictor()

    print_header()

    while True:

        text = input("\nMasukkan tweet:\n> ").strip()

        if text == "":
            print("⚠️ Tweet tidak boleh kosong!")
            continue

        result = predictor.predict(text)

        print_result(result)

        ulang = input("\nIngin mencoba lagi? (y/n): ").strip().lower()

        if ulang != "y":
            print("\nTerima kasih telah menggunakan aplikasi 😊")
            break


if __name__ == "__main__":
    main()