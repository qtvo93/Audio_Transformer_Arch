DROP TABLE IF EXISTS saved_model;
CREATE TABLE saved_model
(
    model_id integer NOT NULL,
    model_name VARCHAR(256) NOT NULL,
    model_file bytea NOT NULL
);