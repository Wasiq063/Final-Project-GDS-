// ================================================================================
// CS343 Graph Data Science — Milestone 3: Graph Machine Learning (GML)
// Dataset  : Crime Investigation (POLE) — Manchester, UK (August 2017)
//            Source: https://github.com/neo4j-graph-examples/pole
// Team     : Muhammad Wasiq Shaikh (ms09205), Zainab Huzefa (zh08968)
// GDS Ver  : 2026.03.0
// ================================================================================
//
// This script implements three GML tasks on the POLE crime dataset:
//
//   GML-1 : Community Detection  — Louvain algorithm to find clusters of
//                                  densely connected persons (criminal networks)
//
//   GML-2 : Node Classification  — Predict whether a Person is a repeat
//                                  offender using graph structure as features.
//                                  Pipeline: degree + FastRP → Random Forest
//                                  with class weighting to handle imbalance
//
//   GML-3 : Link Prediction      — Predict hidden/missing KNOWS relationships
//                                  between Persons who are likely connected
//                                  but undocumented in the dataset.
//                                  Pipeline: FastRP → Hadamard → Random Forest
//
// EXECUTION ORDER
// ───────────────
// Run queries ONE AT A TIME in Neo4j Browser, top to bottom.
// GML-1 MUST complete before GML-3 because GML-3's cross-community query
// uses the communityId property that GML-1 writes to every Person node.
//
// VERIFY GDS IS WORKING — run this first:
RETURN gds.version() AS gdsVersion;


// ================================================================================
// GML-1: COMMUNITY DETECTION — LOUVAIN ALGORITHM
// ================================================================================
//
// Persons in the POLE dataset are connected through five social relationship
// types: KNOWS, FAMILY_REL, KNOWS_PHONE, KNOWS_LW, and KNOWS_SN.
// Louvain maximises "modularity" — how much denser connections are within
// communities compared to random chance — to find natural criminal clusters
// without needing us to specify how many communities to find in advance.
//
// WRITE MODE: communityId is written permanently to every Person node in the
// database. It persists after the projection is dropped and is used in GML-3.


// ── GML-1, Step 1: Create in-memory projection ────────────────────────────────
// UNDIRECTED: social links are the same regardless of stored direction.
// Record nodeCount and relationshipCount for your paper.

CALL gds.graph.project(
    'pole-person-social',
    'Person',
    {
        KNOWS:       { orientation: 'UNDIRECTED' },
        FAMILY_REL:  { orientation: 'UNDIRECTED' },
        KNOWS_PHONE: { orientation: 'UNDIRECTED' },
        KNOWS_LW:    { orientation: 'UNDIRECTED' },
        KNOWS_SN:    { orientation: 'UNDIRECTED' }
    }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;


// ── GML-1, Step 2: Run Louvain ────────────────────────────────────────────────
// Writes communityId to every Person node.
// communityCount = total clusters found
// modularity     = quality score; above 0.3 = meaningful, above 0.6 = strong

CALL gds.louvain.write(
    'pole-person-social',
    {
        writeProperty:  'communityId',
        maxLevels:      10,
        maxIterations:  10,
        tolerance: 0.0001
    }
)
YIELD communityCount, modularity
RETURN communityCount, modularity;


// ── GML-1, Step 3: View community sizes ──────────────────────────────────────
// Largest communities first. Expect a power-law pattern: a few large clusters
// and many singletons — this is normal for real social networks.

MATCH (p:Person)
WHERE p.communityId IS NOT NULL
WITH p.communityId AS community, count(p) AS memberCount
ORDER BY memberCount DESC
RETURN community, memberCount
LIMIT 20;


// ── GML-1, Step 4: Characterise the largest community ────────────────────────
// Lists every member of the biggest cluster and how many crimes they are
// party to. High crime involvement in one community = candidate criminal network.

MATCH (p:Person)
WHERE p.communityId IS NOT NULL
WITH p.communityId AS topCommunity, count(p) AS sz
ORDER BY sz DESC
LIMIT 1

MATCH (p:Person {communityId: topCommunity})
OPTIONAL MATCH (p)-[:PARTY_TO]->(c:Crime)
RETURN
    p.name + ' ' + p.surname AS person,
    topCommunity             AS communityId,
    count(c)                 AS crimesInvolved
ORDER BY crimesInvolved DESC;


// ── GML-1, Step 5: Rank communities by crime density ─────────────────────────
// Size alone is not enough — a small community where everyone has many crimes
// is more suspicious than a large community of innocent people.

MATCH (p:Person)-[:PARTY_TO]->(c:Crime)
WHERE p.communityId IS NOT NULL
WITH p.communityId AS community, count(c) AS totalCrimes
RETURN community, totalCrimes
ORDER BY totalCrimes DESC
LIMIT 20;


// ── GML-1, Step 6: Drop projection ───────────────────────────────────────────
// communityId stays written on every Person node in the database.

CALL gds.graph.drop('pole-person-social')
YIELD graphName
RETURN graphName + ' dropped. communityId saved on all Person nodes.' AS status;


// ================================================================================
// GML-2: NODE CLASSIFICATION — REPEAT OFFENDER (FIXED HYBRID VERSION)
// ================================================================================



// ─────────────────────────────────────────────
// STEP 1: CREATE LABEL 
// ─────────────────────────────────────────────

MATCH (p:Person)
SET p.isRepeatOffender = 0;

MATCH (p:Person)
WITH p, COUNT { (p)-[:PARTY_TO]->(:Crime) } AS crimeCount
WHERE crimeCount > 1
SET p.isRepeatOffender = 1;



// ─────────────────────────────────────────────
// STEP 3: CREATE GRAPH PROJECTION
// ─────────────────────────────────────────────

CALL gds.graph.project(
    'pole-classification',
    {
        Person: { properties: ['isRepeatOffender'] }, 
        Crime: {}
    },
    {
        PARTY_TO: { orientation: 'UNDIRECTED' },
        KNOWS: { orientation: 'UNDIRECTED' },
        FAMILY_REL: { orientation: 'UNDIRECTED' }
    }
);


// ─────────────────────────────────────────────
// STEP 4: CREATE PIPELINE
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.create('repeat-offender-pipeline')
YIELD name;



// ─────────────────────────────────────────────
// STEP 5: DEGREE FEATURE
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.addNodeProperty(
    'repeat-offender-pipeline',
    'degree',
    { mutateProperty: 'degree' }
)
YIELD name;



// ─────────────────────────────────────────────
// STEP 6: FASTRP EMBEDDING (STRONG STRUCTURAL SIGNAL)
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.addNodeProperty(
    'repeat-offender-pipeline',
    'fastRP',
    {
        mutateProperty: 'embedding',
        embeddingDimension: 64,
        randomSeed: 42
    }
)
YIELD name;



// ─────────────────────────────────────────────
// STEP 7: FEATURE SELECTION (FINAL SET)
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.selectFeatures(
    'repeat-offender-pipeline', 
    ['degree', 'embedding']
);


// ─────────────────────────────────────────────
// STEP 8: TRAIN/TEST SPLIT
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.configureSplit(
    'repeat-offender-pipeline',
    {
        testFraction: 0.25,
        validationFolds: 5
    }
)
YIELD splitConfig;



// ─────────────────────────────────────────────
// STEP 9: RANDOM FOREST MODEL
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.addRandomForest(
    'repeat-offender-pipeline',
    {
        numberOfDecisionTrees: 150
    }
)
YIELD parameterSpace;



// ─────────────────────────────────────────────
// STEP 10: TRAIN MODEL
// ─────────────────────────────────────────────

CALL gds.beta.pipeline.nodeClassification.train(
    'pole-classification',
    {
        pipeline: 'repeat-offender-pipeline',
        modelName: 'repeat-offender-model-v3',
        targetNodeLabels: ['Person'],
        targetProperty: 'isRepeatOffender',
        metrics: ['F1_WEIGHTED', 'F1_MACRO'],
        randomSeed: 42
    }
)
YIELD modelInfo
RETURN
    modelInfo.metrics.F1_MACRO.test AS f1Macro,
    modelInfo.metrics.F1_WEIGHTED.test AS f1Weighted;



// ─────────────────────────────────────────────
// STEP 11: WRITE PREDICTIONS
// ─────────────────────────────────────────────

CALL gds.degree.write('pole-classification', { writeProperty: 'degree' });

CALL gds.beta.pipeline.nodeClassification.predict.write(
    'pole-classification',
    {
        modelName: 'repeat-offender-model-v3',
        writeProperty: 'predictedRepeatOffender',
        predictedProbabilityProperty: 'repeatOffenderProbability'
    }
);



// ─────────────────────────────────────────────
// STEP 12: EVALUATION
// ─────────────────────────────────────────────

MATCH (p:Person)
WHERE p.repeatOffenderProbability IS NOT NULL
RETURN
    p.name + ' ' + p.surname AS person,
    p.crimeCount AS crimeCount,
    p.degree AS degree,
    p.repeatOffenderProbability[1] AS probability,
    p.predictedRepeatOffender AS predicted,
    p.isRepeatOffender AS actual
ORDER BY probability DESC
LIMIT 20;

// ── GML-3, Step 13: Table in report to show top TN and FN ───────────────────────────
MATCH (p:Person)
WHERE p.isRepeatOffender = 1
WITH p,
     COUNT { (p)-[:PARTY_TO]->(:Crime) } AS crimes,
     CASE 
         WHEN p.repeatOffenderProbability IS NOT NULL 
         THEN round(p.repeatOffenderProbability[1], 3) 
         ELSE null 
     END AS probability,
     CASE 
         WHEN p.predictedRepeatOffender = 1 THEN 'TP'
         ELSE 'FN'
     END AS result
RETURN
    p.name + ' ' + p.surname AS person,
    crimes,
    p.degree AS degree,
    probability,
    result
ORDER BY probability DESC;

// ================================================================================
// GML-3: LINK PREDICTION — FINDING HIDDEN CONNECTIONS BETWEEN PERSONS
// ================================================================================
//
// 586 documented KNOWS edges exist across 369 persons.
// Link prediction asks: which unconnected pairs are most likely actually connected
// but undocumented — missed by investigators or hidden intentionally?
//
// The model learns from existing KNOWS edges (positive examples) and randomly
// sampled non-existing pairs (negative examples), then scores all unconnected pairs.
//
// DATA SPLIT (586 KNOWS edges):
//   Test set    = 117 edges (20%) — hidden from model during training
//   Train set   = ~47 edges (10% of remainder) + 47 negative samples
//   Feature set = ~422 edges — used only to compute FastRP embeddings
//   (The feature set is kept separate to prevent data leakage — if test edges
//   were used to compute embeddings, the model would "cheat" by seeing answers)
//
// HADAMARD PRODUCT:
//   For link prediction we need one feature vector for each PAIR of nodes.
//   Hadamard multiplies both persons' embedding vectors element-wise.
//   High output = both persons occupy similar positions in the criminal network.




// ── GML-3, Step 1: Create the Person-Person projection ───────────────────────

// All five social relationship types, undirected.
// Link prediction in GDS requires UNDIRECTED orientation.
CALL gds.graph.drop('pole-link-prediction', false);
CALL gds.pipeline.drop('link-prediction-pipeline', false);
CALL gds.model.drop('link-prediction-model', false);


CALL gds.graph.project(
    'pole-link-prediction',
    'Person',
    {
        KNOWS:       { orientation: 'UNDIRECTED' },
        FAMILY_REL:  { orientation: 'UNDIRECTED' },
        KNOWS_PHONE: { orientation: 'UNDIRECTED' },
        KNOWS_LW:    { orientation: 'UNDIRECTED' },
        KNOWS_SN:    { orientation: 'UNDIRECTED' }
    }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;


// ── GML-3, Step 2: Create the pipeline ───────────────────────────────────────

CALL gds.beta.pipeline.linkPrediction.create('link-prediction-pipeline')
YIELD name
RETURN name + ' pipeline created' AS status;


// ── GML-3, Step 3: Add FastRP embedding ──────────────────────────────────────
// Stored as 'lpEmbedding' (separate name from GML-2's 'embedding').
// Same 64-dimensional neighbourhood vector concept.

CALL gds.beta.pipeline.linkPrediction.addNodeProperty(
    'link-prediction-pipeline',
    'fastRP',
    {
        mutateProperty:        'lpEmbedding',
        embeddingDimension:    64,
        iterationWeights:      [0.0, 1.0, 1.0, 1.0, 1.0],
        randomSeed:            42
    }
)
YIELD name, nodePropertySteps
RETURN name, nodePropertySteps;


// ── GML-3, Step 4: Add Hadamard edge feature ─────────────────────────────────
// Element-wise multiplies both endpoint lpEmbedding vectors into one edge vector.

CALL gds.beta.pipeline.linkPrediction.addFeature(
    'link-prediction-pipeline',
    'hadamard',
    { nodeProperties: ['lpEmbedding'] }
)
YIELD featureSteps
RETURN featureSteps;


// ── GML-3, Step 5: Configure train/test split ────────────────────────────────
// testFraction: 0.2          — 20% of KNOWS edges reserved for testing
// trainFraction: 0.6         — 60% of remainder used for training
// negativeSamplingRatio: 1.0 — one random non-existing pair per real KNOWS edge
// validationFolds: 5         — 5-fold cross-validation during training

CALL gds.beta.pipeline.linkPrediction.configureSplit(
    'link-prediction-pipeline',
    {
        testFraction:          0.25,
        trainFraction:         0.6,
        negativeSamplingRatio: 1.0,
        validationFolds:       5
    }
)
YIELD splitConfig
RETURN splitConfig;


// ── GML-3, Step 8: Add Random Forest model ───────────────────────────────────

CALL gds.beta.pipeline.linkPrediction.addRandomForest(
    'link-prediction-pipeline',
    { numberOfDecisionTrees: 100,
    maxDepth: 15,
    minLeafSize: 1}
)
YIELD parameterSpace
RETURN parameterSpace;


// ── GML-3, Step 9: TRAIN THE MODEL ───────────────────────────────────────────
// targetRelationshipType: 'KNOWS' — learns specifically from KNOWS edges.
// Other relationship types provide structural context via FastRP embeddings
// but are not the prediction target.
//
// AUCPR is the primary metric:
//   AUROC can look high just because the model identifies obvious non-edges.
//   AUCPR rewards finding actual hidden connections — what we care about.

CALL gds.model.drop('link-prediction-model', false);

CALL gds.beta.pipeline.linkPrediction.train(
    'pole-link-prediction',
    {
        pipeline: 'link-prediction-pipeline',
        modelName: 'link-prediction-model',
        // 'AUROC' is not used as it's not a valid LinkMetric constant
        metrics: ['AUCPR'], 
        targetRelationshipType: 'KNOWS',
        randomSeed: 42
    }
)
YIELD modelInfo
RETURN
    modelInfo.bestParameters AS bestParams,
    modelInfo.metrics.AUCPR.test AS testAUCPR;
    
// ── GML-3, Step 10: Predict and write top missing links ───────────────────────
// predict.stream returns rows of (node1, node2, probability).
// MERGE creates PREDICTED_KNOWS relationships in the database.
// gds.util.asNode() converts internal GDS node IDs back to real Neo4j nodes.
//
// topN: 100       — top 100 predictions, proportionate to 586 existing edges
// threshold: 0.5  — only write predictions where model confidence > 50%
// sampleRate removed — with only 369 nodes (67,896 possible pairs) GDS can
//                      evaluate all pairs exactly without approximation

CALL gds.beta.pipeline.linkPrediction.predict.stream(
    'pole-link-prediction',
    {
        modelName: 'link-prediction-model',
        topN:      100,
        threshold: 0.45
    }
)
YIELD node1, node2, probability
WITH gds.util.asNode(node1) AS p1, gds.util.asNode(node2) AS p2, probability
MERGE (p1)-[r:PREDICTED_KNOWS]->(p2)
SET r.score = probability
RETURN count(*) AS predictedLinksWritten;

// ── STEP: PERFORMANCE ANALYSIS (Precision@K) ─────────────────────────────
MATCH (p1:Person)-[r:PREDICTED_KNOWS]->(p2:Person)
WITH r.score AS score,
     EXISTS { (p1)-[:PARTY_TO]->(:Crime) } OR EXISTS { (p2)-[:PARTY_TO]->(:Crime) } AS criminalConnection
RETURN
  count(*) AS totalPredicted,
  sum(CASE WHEN criminalConnection THEN 1 ELSE 0 END) AS relevantLinks,
  round(toFloat(sum(CASE WHEN criminalConnection THEN 1 ELSE 0 END)) / count(*), 4) AS precisionAtK;

// ── GML-3, Step 11: Inspect top predicted hidden connections ──────────────────
// High confidence + both persons criminally active = strongest investigative lead.

MATCH (p1:Person)-[r:PREDICTED_KNOWS]->(p2:Person)
WITH p1, p2, r.score AS confidence
ORDER BY confidence DESC
LIMIT 20
OPTIONAL MATCH (p1)-[:PARTY_TO]->(c1:Crime)
WITH p1, p2, confidence, count(c1) AS p1Crimes
OPTIONAL MATCH (p2)-[:PARTY_TO]->(c2:Crime)
RETURN
    p1.name + ' ' + p1.surname AS person1,
    p2.name + ' ' + p2.surname AS person2,
    round(confidence, 4)       AS confidence,
    p1Crimes                   AS person1Crimes,
    count(c2)                  AS person2Crimes
ORDER BY confidence DESC;


// ── GML-3, Step 12: Cross-community predicted links ───────────────────────────
// Requires GML-1 to have run first (communityId must exist on Person nodes).
//
// Finds predicted connections where the two persons are in DIFFERENT Louvain
// communities. These are candidate hidden inter-network bridges — connections
// between separate criminal groups that no investigator has documented.
// This is the combined insight that GML-1 and GML-3 together produce and
// neither task could produce alone. Include top rows in your paper.

MATCH (p1:Person)-[r:PREDICTED_KNOWS]->(p2:Person)
WHERE p1.communityId IS NOT NULL
  AND p2.communityId IS NOT NULL
  AND p1.communityId <> p2.communityId
RETURN
    p1.name + ' ' + p1.surname AS person1,
    p2.name + ' ' + p2.surname AS person2,
    p1.communityId             AS community1,
    p2.communityId             AS community2,
    round(r.score, 4)          AS confidence
ORDER BY confidence DESC
LIMIT 20;



// ── GML-3, Step 13: Drop the projection ──────────────────────────────────────

CALL gds.graph.drop('pole-link-prediction')
YIELD graphName
RETURN graphName + ' dropped.' AS status;


