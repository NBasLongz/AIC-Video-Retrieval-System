import { useCallback, useState } from "react";
import { searchFrames } from "../api/retrievalApi";
import type { RetrievalResult, SearchPayload } from "../types/retrieval.types";

export function useRetrievalSearch() {
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async (payload: SearchPayload, ocrHints: string[]) => {
    setIsSearching(true);
    setError(null);
    try {
      const nextResults = await searchFrames(payload, ocrHints);
      setResults(nextResults);
      return nextResults;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Search failed";
      setError(message);
      setResults([]);
      return [];
    } finally {
      setIsSearching(false);
    }
  }, []);

  return { results, isSearching, error, search, setResults };
}

