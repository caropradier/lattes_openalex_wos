---publications
EXPORT DATA OPTIONS(
 uri='gs://user_bucket_carolina_pradier/artigos_periodicos_*.parquet',
 format='PARQUET',
 overwrite=true) AS
SELECT IDLattes, Nome, Titulo, Ano, DOI, Veiculo, ISSN, Quantidadeautores 
FROM `insyspo.projectdb_lattes_2025_umontreal.PublicacoesArtigosperiodicos` 
WHERE Titulo IS NOT NULL AND Titulo != '' 
    AND SAFE_CAST(Ano AS INT64) IS NOT NULL 
    AND CAST(Ano AS INT64) >= 2008 

---terminal
--gsutil cp -r gs://user_bucket_carolina_pradier/ ~/Downloads/  
--gsutil rm 'gs://user_bucket_carolina_pradier/artigos_periodicos_*.parquet'

----check openalex interlap (doi)

EXPORT DATA OPTIONS(
 uri='gs://user_bucket_carolina_pradier/overlap_oa_lattes_*.parquet',
 format='PARQUET',
 overwrite=true) AS
 WITH primary_topic AS (
  SELECT
    wt.work_id,
    wt.topic_id,
    wt.score,
    t.display_name,
    ROW_NUMBER() OVER (PARTITION BY wt.work_id ORDER BY wt.score DESC) AS rn
  FROM `insyspo.publicdb_openalex_2025_03_rm.works_topics` wt
  JOIN `insyspo.publicdb_openalex_2025_03_rm.topics` t
    ON wt.topic_id = t.id
)
SELECT distinct pl.DOI, OA.id as openalex_id,d.display_name AS domain
FROM `insyspo.projectdb_lattes_2025_umontreal.PublicacoesArtigosperiodicos` as pl
RIGHT JOIN `insyspo.publicdb_openalex_2025_03_rm.works` AS OA 
  ON LOWER(OA.doi) = LOWER(pl.DOI)
LEFT JOIN primary_topic pt
  ON OA.id = pt.work_id AND pt.rn = 1
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.topics` AS t
  ON pt.topic_id = t.id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.domains` as d 
    ON t.domain = d.id
WHERE pl.Titulo IS NOT NULL AND pl.Titulo != '' 
    AND SAFE_CAST(pl.Ano AS INT64) IS NOT NULL 
    AND CAST(pl.Ano AS INT64) >= 2008 
    AND pl.DOI IS NOT NULL
    AND pl.DOI != ''
    AND OA.doi IS not NULL AND OA.doi != ''

---terminal
--gsutil -m cp -rn gs://user_bucket_carolina_pradier/ ~/Downloads/  
--gsutil -m rm 'gs://user_bucket_carolina_pradier/overlap_oa_lattes_*.parquet'

----check openalex interlap (title)

EXPORT DATA OPTIONS(
 uri='gs://user_bucket_carolina_pradier/overlap_oa_lattes_title_*.parquet',
 format='PARQUET',
 overwrite=true) AS
SELECT distinct pl.Titulo, pl.Ano, OA.id as openalex_id,d.display_name AS domain
FROM `insyspo.projectdb_lattes_2025_umontreal.PublicacoesArtigosperiodicos` as pl
RIGHT JOIN `insyspo.publicdb_openalex_2025_03_rm.works` AS OA ON OA.title = pl.Titulo and OA.publication_year = CAST(pl.Ano AS INT64)
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.works_topics` AS wt
  ON OA.id = wt.work_id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.topics` AS t
  ON wt.topic_id = t.id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.domains` as d 
    ON t.domain = d.id
WHERE pl.Titulo IS NOT NULL AND pl.Titulo != '' 
    AND SAFE_CAST(pl.Ano AS INT64) IS NOT NULL 
    AND CAST(pl.Ano AS INT64) >= 2008 


---terminal
--gsutil -m cp -rn gs://user_bucket_carolina_pradier/ ~/Downloads/  
--gsutil -m rm 'gs://user_bucket_carolina_pradier/overlap_oa_lattes_title_*.parquet'


----------------------------opposite exercise: get openalex articles not in lattes

EXPORT DATA OPTIONS(
 uri='gs://user_bucket_carolina_pradier/oa_not_lattes_*.parquet',
 format='PARQUET',
 overwrite=true) AS
 WITH primary_topic AS (
  SELECT
    wt.work_id,
    wt.topic_id,
    wt.score,
    t.display_name,
    ROW_NUMBER() OVER (PARTITION BY wt.work_id ORDER BY wt.score DESC) AS rn
  FROM `insyspo.publicdb_openalex_2025_03_rm.works_topics` wt
  JOIN `insyspo.publicdb_openalex_2025_03_rm.topics` t
    ON wt.topic_id = t.id
)
SELECT distinct OA.id as openalex_id, OA.doi, d.display_name AS domain, OA.title, 
OA.publication_year, aut.id AS author_id, aut.display_name AS author_name, i.display_name AS institution_name,i.type AS institution_type
FROM  `insyspo.publicdb_openalex_2025_03_rm.works` AS OA 
LEFT JOIN primary_topic pt
  ON OA.id = pt.work_id AND pt.rn = 1
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.topics` AS t
  ON pt.topic_id = t.id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.domains` as d 
    ON t.domain = d.id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.works_authorships` AS wa
  ON OA.id = wa.work_id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.institutions` AS i
  ON wa.institution_id = i.id
LEFT JOIN
  `insyspo.publicdb_openalex_2025_03_rm.authors` AS aut
  ON wa.author_id = aut.id
WHERE OA.publication_year >= 2008 
    AND OA.type = 'article'
    AND OA.doi IS not NULL AND OA.doi != ''
    AND i.country = 'Brazil'

---terminal
--gsutil -m cp -rn gs://user_bucket_carolina_pradier/ ~/Downloads/  
--gsutil -m rm 'gs://user_bucket_carolina_pradier/oa_not_lattes_*.parquet'

-----------------------------------wos---------------------------

----get wos matches (doi)
SELECT DISTINCT 
    l.doi,
    ids.OST_BK,
    d.[ESpecialite],
    d.[EDiscipline]
FROM [BDPradier].[dbo].[doi_list_search_lattes] AS l
RIGHT JOIN [WoS].[dbo].[Identifier] as ids 
 -- Remove double quotes from l.doi for matching
    ON REPLACE(REPLACE(l.doi, '"', ''), '''', '') = ids.[Identifier]
LEFT JOIN [WoS].[pex].[Article] as a 
    ON ids.OST_BK = a.OST_BK
LEFT JOIN [WoS].[pex].[Liste_Discipline] as d 
    on a.Code_Discipline = d.Code_Discipline
WHERE l.doi is not null 
    and REPLACE(REPLACE(l.doi, '"', ''), '''', '') != ''


----get wos matches (title)
SELECT DISTINCT 
    l.Title,
    l.Ano, 
    a.OST_BK,
    d.[ESpecialite],
    d.[EDiscipline]
FROM [BDPradier].[dbo].[title_list_search_wos] AS l
RIGHT JOIN [WoS].[pex].[Article] as a 
    ON l.Title = a.Titre and l.Ano = a.[Annee_Bibliographique]
LEFT JOIN [WoS].[pex].[Liste_Discipline] as d 
    on a.Code_Discipline = d.Code_Discipline
WHERE l.Title IS NOT NULL 
    AND l.Title != ''
    AND l.Ano IS NOT NULL
    AND l.Ano != ''


----------------------------opposite exercise: get wos articles not in lattes

select a.OST_BK,a.Seq_No, MIN(a.Addr_No) as Addr_No
into #MinAddress
from Wos.dbo.Address_Name as a
GROUP BY OST_BK, Seq_No;


SELECT DISTINCT 
    a.OST_BK,
    a.Titre,
    a.[Annee_Bibliographique],
    d.[ESpecialite],
    d.[EDiscipline],
    n.First_Name,
    n.Last_Name,
    adr.[Institution],
    ids.[Identifier]
FROM [WoS].[pex].[Article] as a
LEFT JOIN [WoS].[dbo].[Identifier] as ids 
    ON ids.OST_BK = a.OST_BK
LEFT JOIN [WoS].[pex].[Liste_Discipline] as d 
    on a.Code_Discipline = d.Code_Discipline
INNER JOIN  [WoS].[dbo].Summary_Name as n on n.OST_BK=a.OST_BK -- the table with OST_BK, Seq_No of authors and names
LEFT JOIN BDVincent.dbo.leiden_clusters_2024 as leiden_cluster on 'WOS:'+leiden_cluster.ut = a.UID and leiden_cluster.author_seq=n.Seq_No --add leiden cluster ID, you need access to BDKozlowski or BDVincent
LEFT JOIN #MinAddress as adr_fix on adr_fix.OST_BK = a.OST_BK and adr_fix.Seq_No = n.Seq_No --add the fixed address seq
LEFT JOIN Wos.pex.Adresse as adr on adr.OST_BK=a.OST_BK and adr.Ordre = adr_fix.Addr_No --for the country, we need the pex.Addresse table
LEFT JOIN Outil.dbo.Liste_Pays as outil on outil.Pays_ISI=adr.Pays --we need the outil tables to match the new Pays col with the old Eregroupement
WHERE a.[Annee_Bibliographique] >= 2008 and adr.Pays = 'BRAZIL'
    AND a.Code_Document = 1
    AND ids.Identifier IS NOT NULL AND ids.Identifier != ''
    and (ids.Type = 'doi' or ids.Type = 'xref_doi' or ids.Type = 'parent_book_doi')


----get author matches in openalex-----

EXPORT DATA OPTIONS(
 uri='gs://user_bucket_carolina_pradier/openalex_lattes_authors_*.parquet',
 format='PARQUET',
 overwrite=true) AS
 SELECT DISTINCT IDLattes, author_id 
 FROM `insyspo.projectdb_lattes_2025_umontreal.match_researchers_lattes_openalex` 

 ---terminal
--gsutil -m cp -rn gs://user_bucket_carolina_pradier/ ~/Downloads/  
--gsutil -m rm 'gs://user_bucket_carolina_pradier/openalex_lattes_authors_*.parquet'