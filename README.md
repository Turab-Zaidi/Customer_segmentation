
# Customer Segmentation for Targeted Marketing

This project presents an end-to-end unsupervised machine learning pipeline to segment credit card customers into distinct behavioral groups. By leveraging clustering algorithms and dimensionality reduction, this analysis provides actionable business intelligence for designing targeted marketing strategies, increasing customer engagement, and mitigating credit risk.

## Key Findings: A Tale of Two Algorithms

This project highlights a crucial concept in cluster analysis: different algorithms can reveal different, equally valid stories about the same dataset. The key is understanding what each story is useful for.

### K-Means: The Strategic Segmentation

K-Means was tasked with partitioning the entire customer base into 3 distinct groups. The results provide a powerful, high-level strategic overview:

<img width="1784" height="1332" alt="image" src="https://github.com/user-attachments/assets/d1bfd3bb-c659-4556-83f2-80758539822b" />


*   **Persona 1 - High-Value Spenders:** (Cluster 0) Our most profitable segment, defined by high purchases and high credit limits. Represents **~30%** of the customer base.
    *   **Strategy:** Retain and grow with a premium loyalty program.

*   **Persona 2 - At-Risk Debt Holders:** (Cluster 1) Our highest-risk segment, defined by high balances driven by cash advances. Represents **~35%** of the customer base.
    *   **Strategy:** Mitigate risk with debt consolidation offers.

*   **Persona 3 - Prudent & Low-Risk Users:** (Cluster 2) Our most responsible segment, defined by moderate spending and very low balances. Represents **~35%** of the customer base.
    *   **Strategy:** Encourage top-of-wallet usage with targeted cashback rewards.

**Conclusion for K-Means:** This segmentation is ideal for **strategic planning**. It divides the entire market into manageable personas, allowing marketing and risk teams to allocate their budgets and design broad campaigns effectively.

### DBSCAN: The Diagnostic & Tactical Segmentation

DBSCAN was not given a target number of clusters. Instead, it analyzed the density of the data, leading to a drastically different and more literal finding:

<img width="1384" height="984" alt="image" src="https://github.com/user-attachments/assets/f5d60c45-0a3c-43e8-93b2-b2e339257275" />


*   **The "Core" Customer (Cluster 0, ~94% of users):** DBSCAN concluded that the vast majority of customers exist in a single, large, continuous spectrum of behavior rather than in naturally separate groups.
*   **The "True Outliers" (Cluster -1, 489 users):** It successfully isolated a group of 489 customers whose spending habits are so extreme that they don't belong to the core group. **Analysis shows these are the "whale" clients** with significantly higher purchases and credit limits.
*   **Micro-Segments (Clusters 1, 2, 3):** It also identified a few tiny, hyper-specific groups (e.g., 13 users) with nearly identical behaviors, representing niche marketing opportunities.

**Conclusion for DBSCAN:** This segmentation is a powerful **diagnostic tool**. It reveals that forcing a 3-segment strategy (like K-Means did) on a customer base that is largely homogeneous might not be the most efficient approach. Its primary value is tactical: **identifying the high-value outliers and niche micro-segments that K-Means would have missed.**

### Final Recommendation: A Hybrid Strategy

A business should use **both** results:

1.  **Use the K-Means personas (High-Value, At-Risk, Prudent) as the foundation for overall marketing strategy and messaging.**
2.  **Before launching campaigns, use the DBSCAN results to refine targeting:**
    *   **Extract the 489 "Outlier" customers** identified by DBSCAN from all marketing lists. These customers are not "normal" and should be handled by a specialized VIP or risk management team.
    *   **Analyze the micro-segments** for low-cost, targeted experimental campaigns.

This hybrid approach combines the practical, strategic overview of K-Means with the precise, diagnostic power of DBSCAN to create a sophisticated and highly effective customer management strategy.
