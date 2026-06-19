# Building a Semantic Cache for Customer Support: A From-Scratch Walkthrough

## The Problem: Your Support Bot Pays Full Price for the Same Question, Every Time

Customer support traffic is repetitive in a very specific way: the *intents* are a small, closed set (refund status, shipping delay, password reset, business hours), but the *phrasing* is unbounded. "Why hasn't my order shipped yet?", "Where's my package?", and "Why is delivery taking so long?" are the same question wearing three different outfits. An LLM-backed support agent with no cache treats all three as novel: three full completions, three sets of input/output tokens, three round-trips through the model.

A literal cache, keyed on the exact question string, catches almost none of this, because customers essentially never repeat your exact wording back to you. What you actually want is a cache keyed on *meaning*. That's a semantic cache: embed the incoming question, compare it against questions you've already answered, and if one is close enough in vector space, return the stored answer instead of calling the LLM at all.

"Close enough" is the entire mechanism, and it's worth being precise about it up front, because it's easy to mentally simplify a semantic cache into a hit/miss key-value store. It isn't one. There's no equality check anywhere in the lookup, only a distance and a threshold:

![Customer-support query flowing through an embeddings service, a distance-threshold cache check, and RAG](rag-cache-diagram.png)

## Distance, Not Equality

The lookup is two steps: embed the query, then measure how far that embedding sits from every cached question's embedding. "Far" here means cosine distance (`1 - cosine_similarity`), so 0 is identical direction in embedding space and larger values mean less semantically related.

```python
def cosine_dist(a: np.array, b: np.array):
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b) if b.ndim == 1 else np.linalg.norm(b, axis=1)
    sim = np.dot(a, b) / (a_norm * b_norm)
    return 1 - sim

def semantic_search(query: str) -> tuple:
    query_embedding = encoder.encode([query])[0]
    distances = cosine_dist(faq_embeddings, query_embedding)
    best_idx = int(np.argmin(distances))
    return best_idx, distances[best_idx]
```

A cache hit is then just: is the *nearest* cached question within a tolerance `τ`?

```python
def check_cache(query: str, distance_threshold: float = 0.3):
    idx, distance = semantic_search(query)
    if distance <= distance_threshold:
        return {
            "prompt": faq_df.iloc[idx]["question"],
            "response": faq_df.iloc[idx]["answer"],
            "vector_distance": float(distance),
        }
    return None  # cache miss
```

`τ` is the whole design decision. Set it too tight and the cache only ever fires on near-identical phrasing; you've built an expensive way to do nothing. Set it too loose and the cache will confidently return a stored answer to a question it doesn't actually match, with no LLM in the loop to catch the mistake. A miss just costs you a normal LLM call; a *bad hit* costs you correctness, silently. Threshold tuning is a precision/recall trade-off like any other retrieval problem, not a constant you set once.

## Running It For Real

To see this with real numbers rather than asserted ones, I put together a small demo for my own understanding. It runs a distance-threshold check with `τ = 0.3` against a small synthetic FAQ set, using a lightweight local encoder (`all-MiniLM-L6-v2`, a smaller stand-in for `all-mpnet-base-v2`, chosen so the whole thing runs in seconds):

| Query | Closest cached question | Distance | Verdict |
|---|---|---|---|
| "Is it possible to get a refund?" | "How long will it take to get a refund for my order?" | 0.328 | <strong style="color:#1E47E6">MISS</strong> |
| "I want my money back" | "What is your return policy?" | 0.558 | MISS |
| "Why hasn't my order shipped yet?" | "Where is my order, has it shipped yet?" | 0.242 | <strong style="color:#1E47E6">HIT</strong> |
| "I forgot my password, how do I get back in?" | "How do I reset my password?" | 0.173 | <strong style="color:#1E47E6">HIT</strong> |
| "Can I speak to a human on the phone?" | "Do you have a mobile app?" | 0.636 | MISS |

The interesting row is the first one. "Is it possible to get a refund?" is, to a human, obviously a refund question, and it still misses, landing at 0.328 against a threshold of 0.3. It isn't a bug; it's the threshold doing exactly what it's supposed to do with this particular encoder. A smaller, general-purpose embedding model places paraphrases further apart than a model trained specifically for retrieval does. Swap in something retrieval-tuned, such as a larger general-purpose model or a model purpose-built for caching like Redis's `langcache-embed-v1`, and a refund paraphrase like this one would typically land inside the threshold instead of just outside it. The embedding model is a second tuning lever sitting right next to the threshold, not a fixed input to it.

## What a Hit Actually Saves You

Run the cache check 30 times back to back once the encoder is warm, and it averages <strong style="color:#1E47E6">~8ms</strong> per check on a CPU. Almost all of that time goes into encoding the incoming query: comparing against the cached set is just a handful of dot products, essentially free at this scale. The one exception is the very first call after process start, which pays a one-time backend warm-up cost, anywhere from tens of milliseconds to over a second in my runs depending on disk-cache state. That's a tax paid once at boot, not on every query.

Compare that to an actual LLM completion. A real round-trip to a hosted model is dominated by autoregressive decoding, generating output tokens one at a time, which routinely runs from several hundred milliseconds to a few seconds depending on model and response length, on top of network latency. A vector comparison against a handful of floats is simply a different class of operation than generating fifty output tokens.

The same gap shows up in cost, just denominated in tokens instead of milliseconds. A cache hit costs zero input tokens and zero output tokens, because the LLM is never invoked. The lever that converts that into real savings is your hit rate, which depends on how repetitive your actual traffic is. FAQ-style customer support is about as repetitive as LLM workloads get, so a meaningful hit rate isn't optimistic; it's the expected case.

## Adding to the Cache

A cache that only ever gets read from is a static FAQ lookup with extra steps. The other half is writing to it: every new question the LLM has to answer becomes a future cache hit.

```python
def add_to_cache(question: str, answer: str):
    global faq_df, faq_embeddings
    new_row = pd.DataFrame({"question": [question], "answer": [answer]})
    faq_df = pd.concat([faq_df, new_row], ignore_index=True)
    new_embedding = encoder.encode([question])
    faq_embeddings = np.vstack([faq_embeddings, new_embedding])
```

This is the update step: every cache miss that gets resolved by the LLM feeds back into the cache, so the *n*-th paraphrase of a question is nearly free even though the first one wasn't. The cache compounds in value as real traffic flows through it.

## What This Doesn't Solve

Everything here matches on the question in isolation, but a real support conversation carries history: "and how long does *that* take?" means nothing without the preceding turn. Caching turn *N*'s answer while ignoring turns 1 through *N-1* would produce confidently wrong hits in any multi-turn flow. A practical way to handle this is to fold recent context into what actually gets embedded, for example by embedding a short rolling summary of the last couple of turns alongside the latest message, or by scoping the cache key to the conversation and intent rather than to the raw question text alone, so two identical-looking questions from two different conversations don't collide with each other's cached answer.
