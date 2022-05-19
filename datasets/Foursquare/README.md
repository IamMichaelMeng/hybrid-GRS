## Foursquare_FGRec

The format of each file is as follows:

- Foursquare_checkins: user_id, poi_id, check-in frequency.

- Fousquare_context_features: context_id, check_in_frequency, weather_datas

- Foursquare_check_ins_tripartite_graph: user_id, poi_id, check-in frequency, context_id

- Foursquare_data_size: number of users, number of pois, number of catgories.

- Foursquare_poi_coos: poi_id, latitude, longitude.

- Fousquare_poi_features: poi_id, poi_category, latitude, longitud, mask_items

- Foursquare_train/test: user_id, poi_id, check-in frequency.

- Foursquare_neighbor_friends: user_id, user_ids...

- Foursquare_social_relations: user_id, user_id

- Foursquare_social_friends: user_id, user_ids...

- Foursquare_poi_categories: poi_id, cat_id

- Foursquare_user_category: user_id, poi_id, cat_id.

- Foursquare_user_home: user_id, latitude, longitude.

- Foursquare_user_vector: user_id, user_Struct2vec_fts.

- Foursquare_user_features: user_id, user_fts.
