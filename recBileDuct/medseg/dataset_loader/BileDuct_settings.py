def get_Bile_split_policy(identifier="standard", cval=0):

    assert cval < 5 and cval >= 0, 'only support five fold cross validation, but got {}'.format(cval)

    test_list = ["002", "010", "012"]

    if identifier == 'standard':
        # 18/3/3 for training and validation and test.
        training_list = ['001', '003', '004', '005', '006', '007', '008', '011', '014', '015', '017', '018', '019',
                         '020', '021', '022', '024', '025']
        validate_list = ['009', '016', '023']

        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': training_list,
            'val': validate_list,
            'test': test_list,
            'unlabelled': [],
            'test+unlabelled': test_list
        }
    else:
        pass