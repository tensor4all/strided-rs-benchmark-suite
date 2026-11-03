use super::*;

#[test]
fn parses_perm_pattern_with_explicit_strides_without_seed() {
    let json = r#"
    {
      "version": 1,
      "index_base": 0,
      "semantics": "out[i0,...,ik] = src[i_perm0,...,i_permk]",
      "data": "deterministic_index_value",
      "patterns": [
        {
          "id": "tn_light_415_24d_scattered_to_colmajor",
          "label": "24D scattered -> col-major",
          "dtype": "f64",
          "shape": [2, 2, 2, 2],
          "perm": [0, 2, 1, 3],
          "src_layout": {
            "kind": "explicit_strides",
            "strides": [1, 2, 8, 4]
          },
          "dst_layout": { "kind": "col_major" },
          "participants": ["naive", "strided_perm", "julia_base", "strided_jl"],
          "notes": "HPTT cannot represent arbitrary source strides."
        }
      ]
    }
    "#;

    let suite = load_pattern_suite_from_str(json).unwrap();

    assert_eq!(suite.patterns.len(), 1);
    assert_eq!(
        suite.patterns[0].id,
        "tn_light_415_24d_scattered_to_colmajor"
    );
    assert_eq!(suite.patterns[0].perm, vec![0, 2, 1, 3]);
    assert_eq!(
        suite.patterns[0].src_layout,
        LayoutPattern::ExplicitStrides {
            strides: vec![1, 2, 8, 4]
        }
    );
    assert!(!suite.patterns[0].participants.contains(&Participant::Hptt));
}

#[test]
fn bundled_patterns_are_valid() {
    let suite = load_pattern_suite_from_str(include_str!("patterns.json")).unwrap();

    assert!(suite.patterns.iter().any(|p| p.id == "transpose_2d_1024"));
    assert!(suite
        .patterns
        .iter()
        .any(|p| p.id == "tn_light_415_24d_scattered_to_colmajor"));
}

#[test]
fn rejects_seed_in_pattern_schema() {
    let json = r#"
    {
      "version": 1,
      "index_base": 0,
      "semantics": "out[i0,...,ik] = src[i_perm0,...,i_permk]",
      "data": "deterministic_index_value",
      "patterns": [
        {
          "id": "bad_seed",
          "label": "bad seed",
          "dtype": "f64",
          "shape": [2, 2],
          "perm": [1, 0],
          "src_layout": { "kind": "col_major" },
          "dst_layout": { "kind": "col_major" },
          "participants": ["naive"],
          "seed": 42
        }
      ]
    }
    "#;

    assert!(load_pattern_suite_from_str(json).is_err());
}
