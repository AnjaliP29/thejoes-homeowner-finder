import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_outreach(row):
    """
    Generate a personalized outreach message for a homeowner
    using Groq LLM.
    """
    value = row.get("property_value", 0)
    try:
        value_str = f"${float(value):,.0f}"
    except:
        value_str = "your property"

    address = row.get("address", "your property")
    if not address or str(address) == "nan":
        address = "your property"

    market   = row.get("market", "your area")
    verified = row.get("airbnb_verified", False)

    if verified:
        context = (
            f"This homeowner already lists their property on Airbnb, "
            f"so they are comfortable with hosting guests."
        )
    else:
        context = (
            f"This is a high-value property in a premium lifestyle market."
        )

    prompt = f"""Write a short warm outreach message to a homeowner 
on behalf of The Joes — a luxury midterm rental network for 30+ night stays.

Property: {address}, {market}
Estimated Value: {value_str}
Context: {context}

Rules:
- 3 to 4 sentences only
- Mention the specific address and market
- Briefly explain The Joes: curated network, 30+ night stays, 
  quality guests, earn income while they travel
- End with a soft call to action
- Do NOT use a subject line or sign-off
- Sound warm and genuine, not salesy
- Under 80 words"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Groq API error: {e}")
        return (
            f"Hi, we came across your property at {address} in "
            f"{market} and think it would be a wonderful fit for "
            f"The Joes network. We connect premium homeowners with "
            f"high-quality guests for 30+ night stays — giving you "
            f"reliable income while you travel. Would love to connect "
            f"if you're open to it."
        )


def enrich_top_leads(df, n=20):
    """
    Generate outreach messages for top N leads.
    """
    print(f"\n── Generating outreach messages for top {n} leads ──")
    top = df.head(n).copy()
    messages = []

    for idx, (i, row) in enumerate(top.iterrows()):
        print(f"  Generating message {idx+1}/{n}...", end="\r")
        msg = generate_outreach(row)
        messages.append(msg)

    top["outreach_message"] = messages
    print(f"\n  Done — {n} messages generated")
    return top


if __name__ == "__main__":
    from data import load_all_data
    from cross_verify import cross_verify
    from score import score_properties

    props, airbnb = load_all_data()
    if not props.empty:
        verified = cross_verify(props, airbnb)
        scored   = score_properties(verified)
        enriched = enrich_top_leads(scored, n=3)

        print("\nSample outreach messages:")
        for _, row in enriched.iterrows():
            print(f"\n{'─'*55}")
            print(f"Property : {row['address']} | {row['market']}")
            print(f"Value    : ${row['property_value']:,.0f}")
            print(f"Verified : {row['airbnb_verified']} | "
                  f"Score: {row['score']}")
            print(f"\nMessage:\n{row['outreach_message']}")